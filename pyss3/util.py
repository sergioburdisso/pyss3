# -*- coding: utf-8 -*-
"""This is a helper module with utility classes and functions."""
from __future__ import print_function
from io import open
from os import listdir, makedirs, path, remove as remove_file
from sys import version_info
from tqdm import tqdm
from math import ceil
from itertools import product
from collections import defaultdict

from numpy import mean, linspace, arange
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np
import unicodedata
import webbrowser
import errno
import json
import re

try:
    from sklearn.metrics import multilabel_confusion_matrix
except ImportError:
    print("\033[93m* Your Scikit-learn version does not include `multilabel_confusion_matrix()`.\n"
          "* Update Scikit-learn in case you want to work with multi-label classification.\033[0m")

    def multilabel_confusion_matrix(*args):
        """Dummy version of multilabel_confusion_matrix."""
        return np.array([])

ENCODING = "utf-8"

REGEX_DATE = re.compile(
    r"(?:\d+([.,]\d+)?[-/\\]\d+([.,]\d+)?[-/\\]\d+([.,]\d+)?)"
)
REGEX_TEMP = re.compile(
    r"(?:\d+([.,]\d+)?\s*(\xc2[\xb0\xba])?\s*[CcFf](?=[^a-zA-Z]))"
)
REGEX_MONEY = re.compile(r"(?:\$\s*\d+([.,]\d+)?)")
REGEX_PERCENT = re.compile(r"(?:\d+([.,]\d+)?\s*%)")
REGEX_NUMBER = re.compile(r"(?:\d+([.,]\d+)?)")
REGEX_DOTS_CHARS = re.compile(r"(?:([(),;:?!=\"/.|<>\[\]]+)|(#(?![a-zA-Z])))")
REGEX_DOTS_CHAINED = re.compile(r"(?:(#[a-zA-Z0-9]+)(\s)*(?=#))")

EVAL_HTML_OUT_FILE = "ss3_model_evaluation[%s].html"
EVAL_HTML_SRC_FOLDER = "resources/evaluation_plot/"
EVAL_CACHE_EXT = ".ss3ev"

STR_ACCURACY, STR_PRECISION = "accuracy", "precision"
STR_RECALL, STR_F1 = "recall", "f1-score"
STR_HAMMING_LOSS, STR_EXACT_MATCH = "hamming-loss", "exact-match"
GLOBAL_METRICS = [STR_ACCURACY, STR_HAMMING_LOSS, STR_EXACT_MATCH]
METRICS = [STR_PRECISION, STR_RECALL, STR_F1]
EXCP_METRICS = GLOBAL_METRICS + ["confusion_matrix", "categories"]
AVGS = ["micro avg", "macro avg", "weighted avg", "samples avg"]

STR_TEST, STR_FOLD = 'test', 'fold'
STR_MOST_PROBABLE = "most-probable"

ERROR_NAM = "'%s' is not a valid metric " + "(excepted: %s)" % (
    ", ".join(["'%s'" % m for m in [STR_ACCURACY] + METRICS])
)
ERROR_NAT = "'%s' is not a valid target "
ERROR_NAT += "(excepted: %s or a category label)" % (", ".join(["'%s'" % a for a in AVGS]))
ERROR_CNE = "the classifier has not been evaluated yet"
ERROR_CNA = ("a classifier has not yet been assigned "
             "(try using `Evaluation.set_classifier(clf)`)")
ERROR_IKV = "`k_fold` argument must be an integer greater than or equal to 2"
ERROR_IKT = "`k_fold` argument must be an integer"
ERROR_INGV = "`n_grams` argument must be a positive integer"
ERROR_IHV = "hyperparameter values must be numbers"

PY2 = version_info[0] == 2
if not PY2:
    basestring = None  # to avoid the Flake8 "F821 undefined name" error

# a more user-friendly alias for numpy.linspace
# to be used along with grid_search
span = linspace
frange = arange


class VERBOSITY:
    """verbosity "enum" constants."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


class Evaluation:
    """
    Evaluation class.

    This class provides the user easy-to-use methods for model evaluation and
    hyperparameter optimization, like, for example, the ``Evaluation.test``,
    ``Evaluation.kfold_cross_validation``, ``Evaluation.grid_search``,
    ``Evaluation.plot`` methods for performing tests, stratified k-fold cross
    validations, grid searches for hyperparameter optimization, and
    visualizing evaluation results using an interactive 3D plot, respectively.

    All the evaluation methods provided by this class internally use a cache
    mechanism in which all the previously computed evaluation will be
    permanently stored for later use. This will prevent the user to waste time
    performing the same evaluation more than once, or, in case of computer
    crashes or power failure during a long evaluation, once relunched, it will
    skip previously computed values and continue the evaluation from the point
    were the crashed happened on.

    Usage:

    >>> from pyss3.util import Evaluation

    For examples usages for the previously mentioned method, read their
    documentation, or do one of the tutorials.
    """

    __cache__ = None
    __cache_file__ = None
    __clf__ = None
    __last_eval_tag__ = None
    __last_eval_method__ = None
    __last_eval_def_cat__ = None

    @staticmethod
    def __kfold2method__(k_fold):
        """Convert the k number to a proper method string."""
        return STR_TEST if not k_fold or k_fold <= 1 else str(k_fold) + '-' + STR_FOLD

    @staticmethod
    def __set_last_evaluation__(tag, method, def_cat):
        """Save the last evaluation tag, method, and default category."""
        Evaluation.__last_eval_tag__ = tag
        Evaluation.__last_eval_method__ = method
        Evaluation.__last_eval_def_cat__ = def_cat

    @staticmethod
    def __get_last_evaluation__():
        """Return the tag, method and default category used in the last evaluation."""
        if Evaluation.__last_eval_tag__:
            return Evaluation.__last_eval_tag__, \
                Evaluation.__last_eval_method__, \
                Evaluation.__last_eval_def_cat__
        elif Evaluation.__cache__:
            tag = list(Evaluation.__cache__)[0]
            method = list(Evaluation.__cache__[tag])[0]
            def_cat = list(Evaluation.__cache__[tag][method])[0]
            return tag, method, def_cat
        else:
            return None, None, None

    @staticmethod
    def __cache_json_hook__(dct):
        """Convert a given dictionary to a RecursiveDefaultDict."""
        c_ddct = RecursiveDefaultDict()
        for key in dct.keys():
            try:
                c_ddct[float(key)] = dct[key]
            except ValueError:
                c_ddct[key] = dct[key]
        return c_ddct

    @staticmethod
    def __cache_load__():
        """Load evaluations from disk."""
        if not Evaluation.__cache__:
            Print.info("loading evaluations from cache")
            clf = Evaluation.__clf__
            empty_cache = False

            if clf:
                try:
                    with open(Evaluation.__cache_file__, "r", encoding=ENCODING) as json_file:
                        Evaluation.__cache__ = json.loads(
                            json_file.read(),
                            object_hook=Evaluation.__cache_json_hook__
                        )
                except IOError:
                    empty_cache = True
            else:
                empty_cache = True

            if empty_cache:
                Print.info("no evaluation results found, creating a new empty cache")
                Evaluation.__cache__ = RecursiveDefaultDict()

    @staticmethod
    def __cache_update__():
        """Save cached values (evaluations) to disk."""
        clf = Evaluation.__clf__
        if not clf:
            return

        Print.info("updating evaluations cache")
        with open(Evaluation.__cache_file__, "w", encoding=ENCODING) as json_file:
            try:  # Python 3
                json_file.write(json.dumps(Evaluation.__cache__))
            except TypeError:  # Python 2
                json_file.write(json.dumps(Evaluation.__cache__).decode(ENCODING))

    @staticmethod
    def __cache_save_result__(
        cache, categories, accuracy, report, conf_matrix, k_fold, i_fold,
        s, l, p, a, hamming_loss=None
    ):
        """Compute, update and finally save evaluation results to disk."""
        rf = round_fix
        s, l = rf(s), rf(l)
        p, a = rf(p), rf(a)

        cache["categories"] = categories

        # if there aren't previous best results, initialize them to -1
        if cache["accuracy"]["best"]["value"] == {}:
            cache["accuracy"]["best"]["value"] = -1
            if hamming_loss is not None:
                cache["hamming-loss"]["best"]["value"] = -1
            for metric, avg in product(METRICS, AVGS):
                if avg in report:  # scikit-learn > 0.20 does not include 'micro avg' in report
                    cache[metric][avg]["best"]["value"] = -1
            for cat in categories:
                for metric in METRICS:
                    cache[metric]["categories"][cat]["best"]["value"] = -1

        # if fold results array is empty, create new ones
        if cache["accuracy"]["fold_values"][s][l][p][a] == {}:
            cache["accuracy"]["fold_values"][s][l][p][a] = [0] * k_fold
            if hamming_loss is not None:
                cache["hamming-loss"]["fold_values"][s][l][p][a] = [0] * k_fold
            cache["confusion_matrix"][s][l][p][a] = [None] * k_fold
            for metric, avg in product(METRICS, AVGS):
                if avg in report:
                    cache[metric][avg]["fold_values"][s][l][p][a] = [0] * k_fold
            for cat in categories:
                for metric in METRICS:
                    cache[metric]["categories"][cat]["fold_values"][s][l][p][a] = [0] * k_fold

        # saving fold results
        cache["accuracy"]["fold_values"][s][l][p][a][i_fold] = rf(accuracy)
        if hamming_loss is not None:
            cache["hamming-loss"]["fold_values"][s][l][p][a][i_fold] = rf(1 - hamming_loss)
        for metric, avg in product(METRICS, AVGS):
            if avg in report:
                cache[metric][avg]["fold_values"][s][l][p][a][i_fold] = rf(report[avg][metric])
        for cat in categories:
            for metric in METRICS:
                cache[metric]["categories"][cat]["fold_values"][s][l][p][a][i_fold] = rf(
                    report[cat][metric]
                )
        cache["confusion_matrix"][s][l][p][a][i_fold] = conf_matrix.tolist()

        # if this is the last fold, compute and store averages and best values
        if i_fold + 1 == k_fold:
            accuracy_avg = rf(mean(cache["accuracy"]["fold_values"][s][l][p][a]))
            cache["accuracy"]["value"][s][l][p][a] = accuracy_avg

            best_acc = cache["accuracy"]["best"]
            if accuracy_avg > best_acc["value"]:
                best_acc["value"] = accuracy_avg
                best_acc["s"], best_acc["l"] = s, l
                best_acc["p"], best_acc["a"] = p, a

            if hamming_loss is not None:
                hamloss_avg = rf(mean(cache["hamming-loss"]["fold_values"][s][l][p][a]))
                cache["hamming-loss"]["value"][s][l][p][a] = hamloss_avg

                best_haml = cache["hamming-loss"]["best"]
                if hamloss_avg > best_haml["value"]:
                    best_haml["value"] = hamloss_avg
                    best_haml["s"], best_haml["l"] = s, l
                    best_haml["p"], best_haml["a"] = p, a

            for metric, avg in product(METRICS, AVGS):
                if avg in report:
                    metric_avg = rf(mean(cache[metric][avg]["fold_values"][s][l][p][a]))
                    cache[metric][avg]["value"][s][l][p][a] = metric_avg

                    best_metric_avg = cache[metric][avg]["best"]
                    if metric_avg > best_metric_avg["value"]:
                        best_metric_avg["value"] = metric_avg
                        best_metric_avg["s"], best_metric_avg["l"] = s, l
                        best_metric_avg["p"], best_metric_avg["a"] = p, a

            for cat in categories:
                for metric in METRICS:
                    metric_cat_avg = rf(mean(
                        cache[metric]["categories"][cat]["fold_values"][s][l][p][a]
                    ))
                    cache[metric]["categories"][cat]["value"][s][l][p][a] = metric_cat_avg

                    best_metric_cat = cache[metric]["categories"][cat]["best"]
                    if metric_cat_avg > best_metric_cat["value"]:
                        best_metric_cat["value"] = metric_cat_avg
                        best_metric_cat["s"], best_metric_cat["l"] = s, l
                        best_metric_cat["p"], best_metric_cat["a"] = p, a

        Evaluation.__cache_update__()

    @staticmethod
    def __cache_is_in__(tag, method, def_cat, s, l, p, a):
        """Return whether this evaluation is already cached."""
        s, l = round_fix(s), round_fix(l)
        p, a = round_fix(p), round_fix(a)
        results = Evaluation.__cache_get_evaluations__(tag, method, def_cat)
        return results["accuracy"]["value"][s][l][p][a] != {}

    @staticmethod
    def __cache_get_test_evaluation__(tag, def_cat, s, l, p, a):
        """Return test results from the cache storage."""
        s, l = round_fix(s), round_fix(l)
        p, a = round_fix(p), round_fix(a)
        if Evaluation.__cache_is_in__(tag, STR_TEST, def_cat, s, l, p, a):
            Print.info("retrieving test evaluation from cache")

            y_true = []
            y_pred = []

            cache = Evaluation.__cache_get_evaluations__(tag, STR_TEST, def_cat)
            cm = np.array(cache["confusion_matrix"][s][l][p][a][0])
            categories = cache["categories"]

            for icat_true, row in enumerate(cm):
                y_true.extend([icat_true] * sum(row))
                for icat_pred in range(len(row)):
                    y_pred.extend([icat_pred] * row[icat_pred])

            return y_true, y_pred, categories
        return None, None, None

    @staticmethod
    def __cache_get_evaluations__(tag, method, def_cat):
        """Given a tag, a method and a default category return cached evaluation results."""
        Evaluation.__cache_load__()
        return Evaluation.__cache__[tag][method][def_cat]

    @staticmethod
    def __cache_remove_lpsa__(c_metric, s, l, p, a, simulate=False, best=True):
        """Remove evaluation results from cache given hyperparameters s, l, p, a."""
        count = 0
        update_best = False
        if best:
            values = c_metric["value"]
            best = c_metric["best"]

            if best["s"] == s or best["l"] == l or \
               best["p"] == p or best["a"] == a:
                update_best = True
        else:
            values = c_metric
        ss = list(values.keys())
        for _s in ss:
            if s is not None and _s != s:
                continue
            ll = list(values[_s].keys())
            for _l in ll:
                if l is not None and _l != l:
                    continue
                pp = list(values[_s][_l].keys())
                for _p in pp:
                    if p is not None and _p != p:
                        continue
                    aa = list(values[_s][_l][_p].keys())
                    for _a in aa:
                        if a is not None and _a != a:
                            continue

                        if not simulate:
                            del values[_s][_l][_p][_a]

                        count += 1

                    if not values[_s][_l][_p]:
                        del values[_s][_l][_p]

                if not values[_s][_l]:
                    del values[_s][_l]

            if not values[_s]:
                del values[_s]

        if update_best and not simulate:
            c_metric["best"] = Evaluation.__get_global_best__(values)

        return count

    @staticmethod
    def __cache_get_default_tag__(clf, x_data, n_grams=None):
        """Create and return a default cache tag."""
        n_grams = n_grams or len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1
        return "%s_%s[%d-data]" % (
            clf.get_name(),
            "words" if n_grams == 1 else "%d-grams" % n_grams,
            len(x_data)
        )

    @staticmethod
    def __cache_remove__(tag, method, def_cat, s, l, p, a, simulate=False):
        """Remove evaluations from the cache storage."""
        Print.verbosity_region_begin(VERBOSITY.QUIET)
        Evaluation.__cache_load__()
        Print.verbosity_region_end()

        cache = Evaluation.__cache__
        count_details = RecursiveDefaultDict()
        count = 0
        tags = list(cache.keys())
        for t in tags:
            if tag and t != tag:
                continue
            methods = list(cache[t].keys())
            for md in methods:
                if method and md != method:
                    continue
                def_cats = list(cache[t][md].keys())
                for dc in def_cats:
                    if def_cat and dc != def_cat:
                        continue
                    c_accuracy = cache[t][md][dc]["accuracy"]
                    count_details[t][md][dc] = Evaluation.__cache_remove_lpsa__(
                        c_accuracy, s, l, p, a, simulate
                    )
                    count += count_details[t][md][dc]

                    if not simulate:
                        Evaluation.__cache_remove_lpsa__(
                            cache[t][md][dc]["confusion_matrix"], s, l, p, a, best=False
                        )

                    if not cache[t][md][dc]["confusion_matrix"]:
                        del cache[t][md][dc]

                    metrics = list(cache[t][md][dc].keys())
                    for metric in metrics:
                        if metric not in EXCP_METRICS:
                            c_metric = cache[t][md][dc][metric]
                            categories = list(c_metric["categories"].keys())
                            for cat in categories:
                                if not simulate:
                                    Evaluation.__cache_remove_lpsa__(
                                        c_metric["categories"][cat], s, l, p, a
                                    )
                            avgs = list(c_metric.keys())
                            for avg in avgs:
                                if avg != "categories":
                                    c_avg = cache[t][md][dc][metric][avg]
                                    if not simulate:
                                        Evaluation.__cache_remove_lpsa__(c_avg, s, l, p, a)

                    if not cache[t][md][dc]:
                        del cache[t][md][dc]

                if not cache[t][md]:
                    del cache[t][md]

            if not cache[t]:
                del cache[t]

        return count, count_details

    @staticmethod
    def __classification_report_k_fold__(
        tag, method, def_cat, s, l, p, a, plot=True,
        metric=None, metric_target='macro avg'
    ):
        """Create the classification report for k-fold validations."""
        Print.verbosity_region_begin(VERBOSITY.VERBOSE, force=True)

        s, l = round_fix(s), round_fix(l)
        p, a = round_fix(p), round_fix(a)

        cache = Evaluation.__cache_get_evaluations__(tag, method, def_cat)
        categories = cache["categories"]

        multilabel = STR_HAMMING_LOSS in cache
        metric = metric or (STR_ACCURACY if not multilabel else STR_HAMMING_LOSS)
        metric = metric if metric != STR_EXACT_MATCH else STR_ACCURACY

        name_width = max(len(cn) for cn in categories)
        width = max(name_width, len(AVGS[-1]))
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(METRICS)

        report = head_fmt.format('', *['avg'] * len(METRICS), width=width)
        report += '\n'
        report += head_fmt.format('', *METRICS, width=width)
        report += '\n\n'

        for cat in categories:
            report += '{:>{width}s} '.format(cat, width=width)
            for mc in METRICS:
                report += ' {:>9.2f}'.format(
                    cache[mc]["categories"][cat]["value"][s][l][p][a]
                )
            report += '\n'
        report += '\n'
        for av in AVGS:
            if av in cache[mc]:
                report += '{:>{width}s} '.format(av, width=width)
                for mc in METRICS:
                    report += ' {:>9.2f}'.format(
                        cache[mc][av]["value"][s][l][p][a]
                    )
                report += '\n'

        report += "\n\n %s: %.3f\n" % (
            Print.style.bold("Avg. %s" % ("Exact Match Ratio" if multilabel else "Accuracy")),
            cache["accuracy"]["value"][s][l][p][a]
        )
        if multilabel:
            report += " %s: %.3f\n" % (
                Print.style.bold("Avg. Hamming Loss"),
                round_fix(1 - cache["hamming-loss"]["value"][s][l][p][a])
            )

        Print.show(report)
        Print.verbosity_region_end()

        if plot and not multilabel:
            Evaluation.__plot_confusion_matrices__(
                cache["confusion_matrix"][s][l][p][a], categories,
                r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
                %
                (s, l, p, a)
            )

        if metric in GLOBAL_METRICS:
            val = cache[metric]["value"][s][l][p][a]
            return val if metric != STR_HAMMING_LOSS else 1 - val
        else:
            if metric not in cache:
                raise KeyError(ERROR_NAM % str(metric))
            if metric_target in cache[metric]:
                return cache[metric][metric_target]["value"][s][l][p][a]
            elif metric_target in cache[metric]["categories"]:
                return cache[metric]["categories"][metric_target]["value"][s][l][p][a]
            raise KeyError(ERROR_NAT % str(metric_target))

    @staticmethod
    def __plot_confusion_matrices__(cms, classes, info='', max_colums=3, multilabel=False):
        """Show and plot the confusion matrices."""
        import matplotlib.pyplot as plt

        categories = classes
        classes = ['no', 'yes'] if multilabel else classes
        n_cms = len(cms)

        rows = int(ceil(n_cms / (max_colums + .0)))
        columns = max_colums if n_cms > max_colums else n_cms

        title = 'Confusion Matri%s' % ('x' if n_cms == 1 else 'ces')

        if info:
            title += "\n(%s)" % info

        fig, _ = plt.subplots(rows, columns, figsize=(8, 8))
        # fig.tight_layout()

        if n_cms > 1:
            fig.suptitle(title + '\n', fontweight="bold")

        for axi, ax in enumerate(fig.axes):
            if axi >= n_cms:
                fig.delaxes(ax)
                continue

            cm = np.array(cms[axi])
            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_yticks(np.arange(cm.shape[0]))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)

            if n_cms == 1:
                ax.figure.colorbar(im, ax=ax)

            if n_cms == 1:
                ax.set_title(title + '\n', fontweight="bold")
            elif multilabel:
                ax.set_title(categories[axi], fontweight="bold")

            if (axi % max_colums) == 0:
                ax.set_ylabel('True', fontweight="bold")
                ax.set_yticklabels(classes)
            else:
                ax.tick_params(labelleft=False)

            if axi + 1 > n_cms - max_colums or multilabel:
                ax.set_xlabel('Predicted', fontweight="bold")
                ax.set_xticklabels(classes)
            else:
                ax.tick_params(labelbottom=False)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        plt.show()
        plt.close()

    @staticmethod
    def __get_global_best__(values):
        """Given a list of evaluations values, return the best one."""
        best = RecursiveDefaultDict()
        best["value"] = -1
        for s in values:
            for l in values[s]:
                for p in values[s][l]:
                    for a in values[s][l][p]:
                        if values[s][l][p][a] > best["value"]:
                            best["value"] = values[s][l][p][a]
                            best["s"] = s
                            best["l"] = l
                            best["p"] = p
                            best["a"] = a
        return best

    @staticmethod
    def __evaluation_result__(
        clf, y_true, y_pred, categories, def_cat, cache=True, method="test",
        tag=None, folder=False, plot=True, k_fold=1, i_fold=0, force_show=False,
        metric=None, metric_target='macro avg'
    ):
        """Compute evaluation results and save them to disk (cache)."""
        import warnings
        from . import STR_UNKNOWN_CATEGORY, IDX_UNKNOWN_CATEGORY, STR_UNKNOWN
        warnings.filterwarnings('ignore')
        if force_show:
            Print.verbosity_region_begin(VERBOSITY.VERBOSE, force=True)

        multilabel = clf.__multilabel__
        metric = metric or (STR_ACCURACY if not multilabel else STR_HAMMING_LOSS)
        metric = metric if metric != STR_EXACT_MATCH else STR_ACCURACY
        hammingloss = None

        if metric == STR_HAMMING_LOSS and not multilabel:
            raise ValueError("the '%s' metric is only allowed in multi-label classification."
                             % STR_HAMMING_LOSS)

        if not multilabel:
            n_cats = len(categories)
            if def_cat == STR_UNKNOWN:
                if categories[-1] != STR_UNKNOWN_CATEGORY:
                    categories = categories + [STR_UNKNOWN_CATEGORY]
                y_pred = [y if y != IDX_UNKNOWN_CATEGORY else n_cats for y in y_pred]
        else:
            y_pred = membership_matrix(clf, y_pred, labels=False)
            y_true = membership_matrix(clf, y_true, labels=False)
            hammingloss = hamming_loss(y_pred, y_true)

        accuracy = accuracy_score(y_pred, y_true)
        Print.show()
        Print.show(
            classification_report(
                y_true, y_pred,
                labels=range(len(categories)), target_names=categories
            )
        )
        if not multilabel:
            Print.show("\n %s: %.3f" % (Print.style.bold("Accuracy"), accuracy))
        else:
            Print.show("\n %s: %.3f" % (Print.style.bold("Exact Match Ratio"), accuracy))
            Print.show(" %s: %.3f" % (Print.style.bold("Hamming Loss"), hammingloss))

        if not multilabel:
            unclassified = None
            if tag and def_cat == STR_UNKNOWN:
                unclassified = sum(map(lambda v: v == n_cats, y_pred))

            if tag and unclassified:
                cat_acc = []
                for cat in clf.get_categories():
                    cat_acc.append((
                        cat,
                        accuracy_score(
                            [
                                clf.get_category_index(cat) if y == n_cats else y
                                for y in y_pred
                            ],
                            y_true
                        )
                    ))

                best_acc = sorted(cat_acc, key=lambda e: -e[1])[0]
                Print.warn(
                    "A better accuracy (%.3f) would be obtained "
                    "with '%s' as the default category"
                    %
                    (best_acc[1], best_acc[0])
                )
                Print.warn(
                    "(Since %d%% of the documents were classified as 'unknown')"
                    %
                    (unclassified * 100.0 / len(y_true))
                )

            Print.show()

        if not multilabel:
            conf_matrix = confusion_matrix(y_true, y_pred)
        else:
            conf_matrix = multilabel_confusion_matrix(y_true, y_pred)

        report = classification_report(
            y_true, y_pred,
            labels=range(len(categories)), target_names=categories,
            output_dict=True
        )

        s, l, p, a = clf.get_hyperparameters()

        if not cache or not Evaluation.__cache_is_in__(tag, method, def_cat, s, l, p, a):
            Evaluation.__cache_save_result__(
                Evaluation.__cache_get_evaluations__(tag, method, def_cat),
                categories, accuracy, report,
                conf_matrix, k_fold, i_fold,
                s, l, p, a, hamming_loss=hammingloss
            )

        if plot:
            Evaluation.__plot_confusion_matrices__(
                [conf_matrix] if not multilabel else conf_matrix, categories,
                r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
                %
                (s, l, p, a), multilabel=multilabel
            )

        warnings.filterwarnings('default')
        if force_show:
            Print.verbosity_region_end()

        if metric == STR_ACCURACY:
            return accuracy
        elif metric == STR_HAMMING_LOSS:
            return hammingloss
        else:
            if metric_target not in report:
                raise KeyError(ERROR_NAT % str(metric_target))
            if metric not in report[metric_target]:
                raise KeyError(ERROR_NAM % str(metric))

            return report[metric_target][metric]

    @staticmethod
    def __grid_search_loop__(
        clf, x_test, y_test, ss, ll, pp, aa, k_fold,
        i_fold, def_cat, tag, categories, cache=True,
        leave_pbar=True, extended_pbar=False, desc_pbar=None, prep=True
    ):
        """Grid search main loop."""
        method = Evaluation.__kfold2method__(k_fold)

        ss = [round_fix(s) for s in list_by_force(ss)]
        ll = [round_fix(l) for l in list_by_force(ll)]
        pp = [round_fix(p) for p in list_by_force(pp)]
        aa = [round_fix(a) for a in list_by_force(aa)]

        slp_list = list(product(ss, ll, pp))
        desc_pbar = desc_pbar or "Grid search"
        progress_bar = tqdm(
            total=len(slp_list) * len(aa), desc=desc_pbar,
            leave=leave_pbar or (method != 'test' and i_fold + 1 >= k_fold)
        )
        progress_desc = tqdm(
            total=0,
            bar_format='{desc}', leave=leave_pbar,
            disable=not extended_pbar
        )

        Print.verbosity_region_begin(VERBOSITY.QUIET)

        for s, l, p in slp_list:
            clf.set_hyperparameters(s, l, p)
            updated = False
            for a in aa:
                if not cache or not Evaluation.__cache_is_in__(
                    tag, method, def_cat, s, l, p, a
                ):
                    if not updated:
                        progress_desc.set_description_str(
                            " Status: [updating model...] "
                            "(s=%.3f; l=%.3f; p=%.3f; a=%.3f)"
                            %
                            (s, l, p, a)
                        )
                        clf.update_values()
                        updated = True

                    clf.set_alpha(a)
                    progress_desc.set_description_str(
                        " Status: [classifying...] "
                        "(s=%.3f; l=%.3f; p=%.3f; a=%.3f)"
                        %
                        (s, l, p, a)
                    )

                    y_pred = clf.predict(
                        x_test, def_cat,
                        labels=False, leave_pbar=False, prep=prep
                    )

                    Evaluation.__evaluation_result__(
                        clf, y_test, y_pred,
                        categories, def_cat,
                        cache, method, tag,
                        plot=False, k_fold=k_fold, i_fold=i_fold
                    )
                else:
                    progress_desc.set_description_str(
                        " Status: [skipping (already cached)...] "
                        "(s=%.3f; l=%.3f; p=%.3f; a=%.3f)"
                        %
                        (s, l, p, a)
                    )
                progress_bar.update(1)
                progress_desc.update(1)

        progress_desc.set_description_str(" Status: [finished]")

        progress_bar.close()
        progress_desc.close()

        Print.verbosity_region_end()

    @staticmethod
    def set_classifier(clf):
        """
        Set the classifier to be evaluated.

        :param clf: the classifier
        :type clf: SS3
        """
        if Evaluation.__clf__ != clf:
            Evaluation.__clf__ = clf
            Evaluation.__cache_file__ = path.join(
                clf.__models_folder__,
                clf.get_name() + EVAL_CACHE_EXT
            )
            try:
                makedirs(clf.__models_folder__)
            except OSError as ose:
                if ose.errno == errno.EEXIST and path.isdir(clf.__models_folder__):
                    pass
                else:
                    raise
            Evaluation.__cache__ = None
            Evaluation.__cache_load__()

    @staticmethod
    def clear_cache(clf=None):
        """
        Wipe out the evaluation cache (for the given classifier).

        :param clf: the classifier (optional)
        :type clf: SS3
        """
        if clf is not None:
            Evaluation.set_classifier(clf)

        Evaluation.__cache__ = None
        clf = Evaluation.__clf__
        if clf:
            if path.exists(Evaluation.__cache_file__):
                remove_file(Evaluation.__cache_file__)

    @staticmethod
    def plot(html_path='./', open_browser=True):
        """
        Open up an interactive 3D plot with the obtained results.

        This 3D plot is opened up in the web browser and shows the results
        obtained from all the performed evaluations up to date. In addition,
        before showing the plot in the browser, this method also creates a
        portable HTML file containing the 3D plot.

        :param html_path: the path in which to store the portable HTML file
                          (default: './')
        :type html_path: str
        :param open_browser: whether to open the HTML in the browser or not
                             (default: True)
        :type open_browser: bool
        :raises: ValueError
        """
        clf = Evaluation.__clf__

        if not clf:
            raise ValueError(ERROR_CNA)

        if not Evaluation.__cache__:
            Print.info("no evaluations to be plotted")
            return False

        pyss3_path = path.dirname(__file__)
        html_src = EVAL_HTML_SRC_FOLDER
        result_html_file = path.join(html_path, EVAL_HTML_OUT_FILE % clf.__name__)
        fout = open(result_html_file, 'w', encoding=ENCODING)
        fhtml = open(
            path.join(pyss3_path, html_src + "model_evaluation.html"),
            'r', encoding=ENCODING
        )

        for line in fhtml.readlines():
            if "plotly.min.js" in line:
                plotly_path = path.join(pyss3_path, html_src + "plotly.min.js")
                with open(plotly_path, 'r', encoding=ENCODING) as fplotly:
                    fout.write(u'    <script type="text/javascript">')
                    fout.write(fplotly.read())
                    fout.write(u'</script>\n')

            elif "angular.min.js" in line:
                angular_path = path.join(pyss3_path, html_src + "angular.min.js")
                with open(angular_path, 'r', encoding=ENCODING) as fangular:
                    fout.write(u'    <script type="text/javascript">')
                    fout.write(fangular.read())
                    fout.write(u'</script>\n')

            elif "data.js" in line:
                fout.write(u'    <script type="text/javascript">')
                fout.write(u'var $model_name = "%s"; ' % clf.get_name())
                fout.write(
                    u'var $results = JSON.parse("%s");'
                    %
                    json.dumps(Evaluation.__cache__).replace('"', r'\"')
                )
                fout.write(u'</script>\n')
            elif "[[__version__]]" in line:
                from pyss3 import __version__
                fout.write(line.replace("[[__version__]]", __version__))
            else:
                fout.write(line)

        fhtml.close()
        fout.close()

        if open_browser:
            webbrowser.open(result_html_file)

        Print.info("evaluation plot saved in '%s'" % result_html_file)
        return True

    @staticmethod
    def get_best_hyperparameters(
        metric=None, metric_target='macro avg', tag=None, method=None, def_cat=None
    ):
        """
        Return the best hyperparameter values for the given metric.

        From all the evaluations performed using the given ``method``, default
        category(``def_cat``) and cache ``tag``, this method returns the
        hyperparameter values that performed the best, according to the given
        ``metric``, if not supplied, these values will automatically use the
        ones matching the last performed evaluation.

        Available metrics are: 'accuracy', 'f1-score', 'precision', and
        'recall'. In addition, In multi-label classification also 'hamming-loss'
        and 'exact-match'

        Except for accuracy, a ``metric_target`` option must also be supplied
        along with the ``metric`` indicating the target we aim at measuring,
        that is, whether we want to measure some averaging performance  or the
        performance on a particular category.

        :param metric: the evaluation metric to return, options are:
                       'accuracy', 'f1-score', 'precision', or 'recall'
                       When working with multi-label classification problems,
                       two more options are allowed: 'hamming-loss' and 'exact-match'.
                       Note: exact match will produce the same result than 'accuracy'.
                       (default: 'accuracy', or 'hamming-loss' for multi-label case).
        :type metric: str
        :param metric_target: the target we aim at measuring with the given
                              metric. Options are: 'macro avg', 'micro avg',
                              'weighted avg' or a category label (default
                              'macro avg').
        :type metric_target: str
        :param tag: the cache tag from where to look up the results
                     (by default it will automatically use the tag of the last
                     evaluation performed)
        :type tag: str
        :param method: the evaluation method used, options are: 'test', 'K-fold',
                       where K is a positive integer (by default it will match the
                       method of the last evaluation performed).
        :type method: str
        :param def_cat: the default category used the evaluations, options are:
                        'most-probable', 'unknown' or a category label (by default
                        it will use the same as the last evaluation performed).
        :type def_cat: str
        :returns: a tuple of the hyperparameter values: (s, l, p, a).
        :rtype: tuple
        :raises: ValueError, LookupError, KeyError
        """
        if not Evaluation.__clf__:
            raise ValueError(ERROR_CNA)

        l_tag, l_method, l_def_cat = Evaluation.__get_last_evaluation__()
        tag, method, def_cat = tag or l_tag, method or l_method, def_cat or l_def_cat
        cache = Evaluation.__cache__[tag][method][def_cat]

        multilabel = STR_HAMMING_LOSS in cache
        metric = metric or (STR_ACCURACY if not multilabel else STR_HAMMING_LOSS)
        metric = metric if metric != STR_EXACT_MATCH else STR_ACCURACY

        if metric not in METRICS + GLOBAL_METRICS:
            raise KeyError(ERROR_NAM % str(metric))

        if metric_target not in AVGS and metric_target not in cache["categories"]:
            raise KeyError(ERROR_NAT % str(metric_target))

        c_metric = cache[metric]

        if metric in GLOBAL_METRICS:
            best = c_metric["best"]
        else:
            if metric_target in AVGS:
                best = c_metric[metric_target]["best"]
            else:
                best = c_metric["categories"][metric_target]["best"]

        if not best:
            raise LookupError(ERROR_CNE)

        return best["s"], best["l"], best["p"], best["a"]

    @staticmethod
    def show_best(tag=None, method=None, def_cat=None, metric=None, avg=None):
        """
        Print information regarding the best obtained values according to all the metrics.

        The information showed can be filtered out using any of the following arguments:

        :param tag: a cache tag (optional).
        :type tag: str
        :param method: an evaluation method, options are: 'test', 'K-fold',
                       where K is a positive integer (optional).
        :type method: str
        :param def_cat: a default category used in the evaluations, options are:
                        'most-probable', 'unknown' or a category label (optional).
        :type def_cat: str
        :param metric: an evaluation metric, options are: 'accuracy', 'f1-score',
                       'precision', and 'recall'. In addition, In multi-label
                        classification also 'hamming-loss' and 'exact-match'
                        (optional).
        :type metric: str
        :param avg: an averaging method, options are: 'macro avg', 'micro avg',
                    and 'weighted avg' (optional).
        :type avg: str
        :raises: ValueError
        """
        if not Evaluation.__clf__:
            raise ValueError(ERROR_CNA)

        cache = Evaluation.__cache__
        ps = Print.style

        if not cache:
            return

        for t in cache:
            if tag and t != tag:
                continue

            for md in cache[t]:
                if method and md != method:
                    continue

                for dc in cache[t][md]:
                    if def_cat and dc != def_cat:
                        continue

                    print("\n%s %s %s %s " % (
                        ps.fail(">"), ps.green(md),
                        ps.blue(t), ps.blue(dc)
                    ), end='')

                    multilabel = STR_HAMMING_LOSS in cache[t][md][dc]

                    evl = cache[t][md][dc]["accuracy"]["value"]
                    n_evl = len([
                        a for s in evl for l in evl[s]
                        for p in evl[s][l] for a in evl[s][l][p]
                    ])
                    print("(%d evaluations)" % n_evl)

                    if multilabel:
                        best = cache[t][md][dc]["hamming-loss"]["best"]
                        print(
                            "    Best %s: %s %s" % (
                                ps.green("hamming loss"),
                                ps.warning(round_fix(1 - best["value"])),
                                ps.blue("(s=%s, l=%s, p=%s, a=%s)") % (
                                    best["s"], best["l"], best["p"], best["a"]
                                )
                            )
                        )

                    best = cache[t][md][dc]["accuracy"]["best"]
                    print(
                        "    Best %s: %s %s" % (
                            ps.green("accuracy" if not multilabel else "exact match ratio"),
                            ps.warning(best["value"]),
                            ps.blue("(s=%s, l=%s, p=%s, a=%s)") % (
                                best["s"], best["l"], best["p"], best["a"]
                            )
                        )
                    )
                    for mc in sorted(cache[t][md][dc].keys()):
                        if metric and mc != metric:
                            continue

                        if mc not in EXCP_METRICS:
                            print("    Best %s:" % ps.green(mc))
                            c_metric = cache[t][md][dc][mc]
                            for cat in c_metric["categories"]:
                                best = c_metric["categories"][cat]["best"]
                                print((" " * 8) + "%s: %s %s" % (
                                    cat, ps.warning(best["value"]),
                                    ps.blue("(s=%s, l=%s, p=%s, a=%s)") % (
                                        best["s"], best["l"], best["p"], best["a"]
                                    )
                                ))

                            print((" " * 8) + ps.header(ps.bold("Averages:")))
                            for av in c_metric:
                                if avg and av != avg:
                                    continue

                                if av != "categories":
                                    best = c_metric[av]["best"]
                                    print((" " * 10) + "%s: %s %s" % (
                                        ps.header(av), ps.warning(best["value"]),
                                        ps.blue("(s=%s, l=%s, p=%s, a=%s)")
                                        % (
                                            best["s"], best["l"],
                                            best["p"], best["a"]
                                        )
                                    ))

        print()

    @staticmethod
    def remove(s=None, l=None, p=None, a=None, method=None, def_cat=None, tag=None, simulate=False):
        """
        Remove evaluation results from the cache storage.

        If not arguments are given, this method will remove everything, and
        thus it will perform exactly like the ``clear_chache`` method.
        However, when arguments are given, only values matching that argument
        value will be removed. For example:

        >>> # remove evaluation results using
        >>> # s=.5, l=1, and p=1 hyperparameter values:
        >>> Evaluation.remove(s=.5, l=1, p=1)
        >>>
        >>> # remove all 4-fold cross validation evaluations:
        >>> Evaluation.remove(method="4-fold")

        Besides, this method returns the number of items that were removed.
        If the argument ``simulate`` is set to ``True``, items won't be removed
        and only the number of items to be removed will be returned. For example:

        >>> c, _ = Evaluation.remove(s=.45, method="test", simulate=True)
        >>> if input("%d items will be removed, proceed? (y/n)") == 'y':
        >>>     Evaluation.remove(s=.45, method="test")

        Here is the full list of arguments that can be used to select what to be
        permanently removed from the cache storage:

        :param s: a value for the s hyperparameter (optional).
        :type s: float
        :param l: a value for the l hyperparameter (optional).
        :type l: float
        :param p: a value for the p hyperparameter (optional).
        :type p: float
        :param a: a value for the a hyperparameter (optional).
        :type a: float
        :param method: an evaluation method, options are: 'test', 'K-fold',
                       where K is a positive integer (optional).
        :type method: str
        :param def_cat: a default category used in the evaluations, options are:
                        'most-probable', 'unknown' or a category label (optional).
        :type def_cat: str
        :param metric: an evaluation metric, options are: 'accuracy', 'f1-score',
                       'precision', and 'recall' (optional).
        :param tag: a cache tag (optional).
        :type tag: str
        :param simulate: whether to simulate the removal or not (default: False)
        :type simulate: bool
        :returns: (number of items removed, details)
        :rtype: tuple
        :raises: ValueError, TypeError
        """
        clf = Evaluation.__clf__
        count, count_details = None, None

        if not clf:
            raise ValueError(ERROR_CNA)

        try:
            if s is not None:
                float(s)
            if l is not None:
                float(l)
            if p is not None:
                float(p)
            if a is not None:
                float(a)
        except (ValueError, TypeError, AttributeError):
            raise TypeError(ERROR_IHV)

        try:
            count, count_details = Evaluation.__cache_remove__(
                tag, method, def_cat, s, l, p, a, simulate
            )
            if Evaluation.__cache__:
                Evaluation.__cache_update__()
            else:
                Evaluation.clear_cache()
        except EOFError:
            pass
        return count, count_details

    @staticmethod
    def test(
        clf, x_test, y_test, def_cat=STR_MOST_PROBABLE, prep=True,
        tag=None, plot=True, metric=None, metric_target='macro avg', cache=True
    ):
        """
        Test the model using the given test set.

        Examples:

        >>> from pyss3.util import Evaluation
        >>> ...
        >>> acc = Evaluation.test(clf, x_test, y_test)
        >>> print("Accuracy:", acc)
        >>> ...
        >>> # this line won't perform the test again, it will retrieve, from the cache storage,
        >>> # the f1-score value computed in previous test.
        >>> f1 = Evaluation.test(clf, x_test, y_test, metric="f1-score")
        >>> print("F1 score:", f1)

        :param clf: the classifier to be evaluated.
        :type clf: SS3
        :param x_test: the test set documents, i.e, the list of documents to be classified
        :type x_test: list (of str)
        :param y_test: the test set with category labels, i.e, the list of document labels
        :type y_test: list (of str) or list (of list of str)
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        'most-probable', 'unknown' or a given category name.
                        (default: 'most-probable')
        :type def_cat: str
        :param prep: enables the default input preprocessing when classifying
                     (default: True)
        :type prep: bool
        :param tag: the cache tag to be used, i.e. a string to identify this evaluation
                    inside the cache storage (optional)
        :type tag: str
        :param plot: whether to plot the confusion matrix after finishing the test or not
                     (default: True)
        :type plot: bool
        :param metric: the evaluation metric to return, options are:
                        'accuracy', 'f1-score', 'precision', or 'recall'
                        When working with multi-label classification problems,
                        two more options are allowed: 'hamming-loss' and 'exact-match'.
                        Note: exact match will produce the same result than 'accuracy'.
                        (default: 'accuracy', or 'hamming-loss' for multi-label case).
        :type metric: str
        :param metric_target: the target we aim at measuring with the given
                              metric. Options are: 'macro avg', 'micro avg',
                              'weighted avg' or a category label (default
                              'macro avg').
        :type metric_target: str
        :param cache: whether to use cached values or not. Setting ``cache=False`` forces
                      to completely perform the evaluation ignoring cached values
                      (default: True).
        :type cache: bool
        :returns: the given metric value, by default, the obtained accuracy.
        :rtype: float
        :raises: EmptyModelError, KeyError, ValueError
        """
        multilabel = clf.__multilabel__
        Evaluation.set_classifier(clf)
        tag = tag or Evaluation.__cache_get_default_tag__(clf, x_test)
        s, l, p, a = clf.get_hyperparameters()

        Evaluation.__set_last_evaluation__(tag, STR_TEST, def_cat)

        y_pred = None
        if cache:
            _y_test, y_pred, categories = Evaluation.__cache_get_test_evaluation__(
                tag, def_cat, s, l, p, a
            )

        # if not cached
        if not y_pred or multilabel:
            clf.set_hyperparameters(s, l, p, a)
            y_pred = clf.predict(x_test, def_cat, prep=prep, labels=False)
            categories = clf.get_categories()
            if not multilabel:
                y_test = [clf.get_category_index(y) for y in y_test]
            else:
                y_test = [[clf.get_category_index(y) for y in yy] for yy in y_test]
        else:
            y_test = _y_test

        return Evaluation.__evaluation_result__(
            clf, y_test, y_pred, categories,
            def_cat, cache, STR_TEST, tag,
            metric=metric, metric_target=metric_target,
            plot=plot, force_show=True
        )

    @staticmethod
    def kfold_cross_validation(
        clf, x_train, y_train, k=4, n_grams=None, def_cat=STR_MOST_PROBABLE, prep=True,
        tag=None, plot=True, metric=None, metric_target='macro avg', cache=True
    ):
        """
        Perform a Stratified k-fold cross validation on the given training set.

        Examples:

        >>> from pyss3.util import Evaluation
        >>> ...
        >>> acc = Evaluation.kfold_cross_validation(clf, x_train, y_train)
        >>> print("Accuracy obtained using (default) 4-fold cross validation:", acc)
        >>> ...
        >>> # this line won't perform the cross validation again, it will retrieve, from the
        >>> #  cache storage, the f1-score value computed in previous evaluation
        >>> f1 = Evaluation.kfold_cross_validation(clf, x_train, y_train, metric="f1-score")
        >>> print("F1 score obtained using (default) 4-fold cross validation:", f1)
        >>> ...
        >>> f1 = Evaluation.kfold_cross_validation(clf, x_train, y_train, k=10, metric="f1-score")
        >>> print("F1 score obtained using 10-fold cross validation:", f1)

        :param clf: the classifier to be evaluated.
        :type clf: SS3
        :param x_train: the list of documents
        :type x_train: list (of str)
        :param y_train: the list of document category labels
        :type y_train: list (of str)
        :param k: indicates the number of folds to be used (default: 4).
        :type k: int
        :param n_grams: indicates the maximum ``n``-grams to be learned
                        (e.g. a value of ``1`` means only 1-grams (words),
                        ``2`` means 1-grams and 2-grams,
                        ``3``, 1-grams, 2-grams and 3-grams, and so on.
        :type n_grams: int
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        'most-probable', 'unknown' or a given category name.
                        (default: 'most-probable')
        :type def_cat: str
        :param prep: enables the default input preprocessing when classifying
                     (default: True)
        :type prep: bool
        :param tag: the cache tag to be used, i.e. a string to identify this evaluation
                    inside the cache storage (optional)
        :type tag: str
        :param plot: whether to plot the confusion matrix after finishing the test or not
                     (default: True)
        :type plot: bool
        :param metric: the evaluation metric to return, options are:
                        'accuracy', 'f1-score', 'precision', or 'recall'
                        When working with multi-label classification problems,
                        two more options are allowed: 'hamming-loss' and 'exact-match'.
                        Note: exact match will produce the same result than 'accuracy'.
                        (default: 'accuracy', or 'hamming-loss' for multi-label case).
        :type metric: str
        :param metric_target: the target we aim at measuring with the given
                              metric. Options are: 'macro avg', 'micro avg',
                              'weighted avg' or a category label (default
                              'macro avg').
        :type metric_target: str
        :param cache: whether to use cached values or not. Setting ``cache=False`` forces
                      to completely perform the evaluation ignoring cached values
                      (default: True).
        :type cache: bool
        :returns: the given metric value, by default, the obtained accuracy.
        :rtype: float
        :raises: InvalidCategoryError, EmptyModelError, ValueError, KeyError
        """
        from . import SS3, EmptyModelError

        if not clf or not clf.__categories__:
            raise EmptyModelError

        if not isinstance(k, int) or k < 2:
            raise ValueError(ERROR_IKV)

        if n_grams is not None and (not isinstance(n_grams, int) or n_grams < 1):
            raise ValueError(ERROR_INGV)

        multilabel = clf.__multilabel__
        kfold_split = MultilabelStratifiedKFold if multilabel else StratifiedKFold

        Evaluation.set_classifier(clf)
        n_grams = n_grams or (len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1)
        tag = tag or Evaluation.__cache_get_default_tag__(clf, x_train, n_grams)

        Print.verbosity_region_begin(VERBOSITY.NORMAL)
        method = Evaluation.__kfold2method__(k)

        Evaluation.__set_last_evaluation__(tag, method, def_cat)

        s, l, p, a = clf.get_hyperparameters()
        categories = clf.get_categories()
        x_train, y_train = np.array(x_train), np.array(y_train)
        skf = kfold_split(n_splits=k)
        pbar_desc = "k-fold validation"
        progress_bar = tqdm(total=k, desc=pbar_desc)
        y_train_split = membership_matrix(clf, y_train).todense() if multilabel else y_train
        for i_fold, (train_ix, test_ix) in enumerate(skf.split(x_train, y_train_split)):
            if not cache or not Evaluation.__cache_is_in__(
                tag, method, def_cat, s, l, p, a
            ):
                x_train_fold, y_train_fold = x_train[train_ix], y_train[train_ix]

                clf_fold = SS3()
                clf_fold.set_hyperparameters(s, l, p, a)
                Print.verbosity_region_begin(VERBOSITY.QUIET)
                progress_bar.set_description_str(pbar_desc + " [training...]")
                clf_fold.fit(x_train_fold, y_train_fold, n_grams,
                             prep=prep, leave_pbar=False)

                x_train_fold, y_train_fold = None, None  # free memory

                if not multilabel:
                    y_test_fold = [clf_fold.get_category_index(y) for y in y_train[test_ix]]
                else:
                    y_test_fold = [[clf_fold.get_category_index(y) for y in yy]
                                   for yy in y_train[test_ix]]
                x_test_fold = x_train[test_ix]

                progress_bar.set_description_str(pbar_desc + " [classifying...]")
                y_pred = clf_fold.predict(x_test_fold, def_cat,
                                          prep=prep, labels=False, leave_pbar=False)
                Print.verbosity_region_end()

                Evaluation.__evaluation_result__(
                    clf_fold, y_test_fold, y_pred,
                    categories, def_cat,
                    cache, method, tag,
                    plot=False, k_fold=k, i_fold=i_fold
                )
                x_test_fold, y_test_fold, y_pred = None, None, None  # free memory

            progress_bar.update(1)

        progress_bar.set_description_str(pbar_desc + " [finished]")
        progress_bar.close()
        Print.verbosity_region_end()

        Print.show()
        return Evaluation.__classification_report_k_fold__(
            tag, method, def_cat, s, l, p, a,
            metric=metric, metric_target=metric_target, plot=plot
        )

    @staticmethod
    def grid_search(
        clf, x_data, y_data, s=None, l=None, p=None, a=None,
        k_fold=None, n_grams=None, def_cat=STR_MOST_PROBABLE, prep=True,
        tag=None, metric=None, metric_target='macro avg', cache=True, extended_pbar=False
    ):
        """
        Perform a grid search using the provided hyperparameter values.

        Given a test or a training set, this method performs a grid search using the given
        lists of hyperparameters values. Once finished, it returns the best hyperparameter
        values found for the given ``metric``.

        If the argument ``k_fold`` is provided, the grid search will perform a
        stratified k-fold cross validation for each hyperparameter value
        combination. If ``k_fold`` is not given, will use the ``x_data`` as if
        it were a test set (x_test) and will use this test set to evaluate the
        classifier performance for each hyperparameter value combination.

        Examples:

        >>> from pyss3.util import Evaluation
        >>> ...
        >>> best_s, _, _, _ = Evaluation.grid_search(clf, x_test, y_test, s=[.3, .4, .5])
        >>> print("For this test set, the value of s that obtained the "
        >>>       "best accuracy, among .3, .4, and .5, was:", best_s)
        >>> ...
        >>> s, l, p, _ = Evaluation.grid_search(clf,
        >>>                                     x_test, y_test,
        >>>                                     s = [.3, .4, .5],
        >>>                                     l = [.5, 1, 1.5],
        >>>                                     p = [.5, 1, 2])
        >>> print("For this test set and these hyperparameter values, "
        >>>       "the value of s, l and p that obtained the best accuracy were, "
        >>>       "respectively:", s, l, p)
        >>> ...
        >>> # since this grid search performs the same computation than the above
        >>> # cached values will be used, instead of computing all over again.
        >>> s, l, p, _ = Evaluation.grid_search(clf,
        >>>                                     x_test, y_test,
        >>>                                     s = [.3, .4, .5],
        >>>                                     l = [.5, 1, 1.5],
        >>>                                     p = [.5, 1, 2],
        >>>                                     metric="f1-score")
        >>> print("For this test set and these hyperparameter values, "
        >>>       "the value of s, l and p that obtained the best F1 score were, "
        >>>       "respectively:", s, l, p)
        >>> ...
        >>> s, l, p, _ = Evaluation.grid_search(clf,
        >>>                                     x_train, y_train,
        >>>                                     s = [.3, .4, .5],
        >>>                                     l = [.5, 1, 1.5],
        >>>                                     p = [.5, 1, 2],
        >>>                                     k_fold=4)
        >>> print("For this training set and these hyperparameter values, "
        >>>       "and using stratified 4-fold cross validation, "
        >>>       "the value of s, l and p that obtained the best accuracy were, "
        >>>       "respectively:", s, l, p)

        :param clf: the classifier to be evaluated.
        :type clf: SS3
        :param x_data: a list of documents
        :type x_data: list (of str)
        :param y_data: a list of document category labels
        :type y_data: list (of str)
        :param s: the list of values for the ``s`` hyperparameter (optional).
                  If not given, will take the classifier (``clf``) current value.
        :param l: the list of values for the ``l`` hyperparameter (optional).
                  If not given, will take the classifier (``clf``) current value.
        :param p: the list of values for the ``p`` hyperparameter (optional).
                  If not given, will take the classifier (``clf``) current value.
        :param a: the list of values for the ``a`` hyperparameter (optional).
                  If not given, will take the classifier (``clf``) current value.
        :param k_fold: indicates the number of folds to be used (optional).
                       If not given, it will perform the grid search using
                       the ``x_data`` as the test test.
        :type k_fold: int
        :param n_grams: indicates the maximum ``n``-grams to be learned
                        (e.g. a value of ``1`` means only 1-grams (words),
                        ``2`` means 1-grams and 2-grams,
                        ``3``, 1-grams, 2-grams and 3-grams, and so on.
        :type n_grams: int
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        'most-probable', 'unknown' or a given category name.
                        (default: 'most-probable')
        :type def_cat: str
        :param prep: enables the default input preprocessing when classifying
                     (default: True)
        :type prep: bool
        :param tag: the cache tag to be used, i.e. a string to identify this evaluation
                    inside the cache storage (optional)
        :type tag: str
        :param metric: the evaluation metric to return, options are:
                        'accuracy', 'f1-score', 'precision', or 'recall'
                        When working with multi-label classification problems,
                        two more options are allowed: 'hamming-loss' and 'exact-match'.
                        Note: exact match will produce the same result than 'accuracy'.
                        (default: 'accuracy', or 'hamming-loss' for multi-label case).
        :type metric: str
        :param metric_target: the target we aim at measuring with the given
                              metric. Options are: 'macro avg', 'micro avg',
                              'weighted avg' or a category label (default
                              'macro avg').
        :type metric_target: str
        :param cache: whether to use cached values or not. Setting ``cache=False`` forces
                      to completely perform the evaluation ignoring cached values
                      (default: True).
        :type cache: bool
        :param extended_pbar: whether to show an extra status bar along with
                              the progress bar (default: False).
        :type extended_pbar: bool
        :returns: a tuple of hyperparameter values (s, l, p, a) with the best
                  values for the given metric
        :rtype: tuple
        :raises: InvalidCategoryError, EmptyModelError, ValueError, TypeError
        """
        if k_fold is not None and not isinstance(k_fold, int):
            raise TypeError(ERROR_IKT)

        from . import SS3
        multilabel = clf.__multilabel__
        Evaluation.set_classifier(clf)
        n_grams = n_grams or (len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1)
        tag = tag or Evaluation.__cache_get_default_tag__(clf, x_data, n_grams)
        method = Evaluation.__kfold2method__(k_fold)

        Evaluation.__set_last_evaluation__(tag, method, def_cat)

        s = s if s is not None else clf.get_s()
        l = l if l is not None else clf.get_l()
        p = p if p is not None else clf.get_p()
        a = a if a is not None else clf.get_a()

        Print.show()
        if not k_fold:  # if test
            if not multilabel:
                x_test, y_test = x_data, [clf.get_category_index(y) for y in y_data]
            else:
                x_test, y_test = x_data, [[clf.get_category_index(y) for y in yy]
                                          for yy in y_data]
            Evaluation.__grid_search_loop__(
                clf, x_test, y_test, s, l, p, a, 1, 0,
                def_cat, tag, clf.get_categories(), cache,
                extended_pbar=extended_pbar, prep=prep
            )
        else:  # if k-fold
            Print.verbosity_region_begin(VERBOSITY.NORMAL)

            x_data, y_data = np.array(x_data), np.array(y_data)
            kfold_split = MultilabelStratifiedKFold if multilabel else StratifiedKFold
            skf = kfold_split(n_splits=k_fold)
            categories = clf.get_categories()

            y_data_split = membership_matrix(clf, y_data).todense() if multilabel else y_data
            k_fold_splits = enumerate(skf.split(x_data, y_data_split))
            for i_fold, (train_ix, test_ix) in k_fold_splits:
                x_train, y_train = x_data[train_ix], y_data[train_ix]

                clf_fold = SS3()
                clf_fold.fit(x_train, y_train, n_grams, prep=prep, leave_pbar=False)

                x_train, y_train = None, None  # free memory

                if not multilabel:
                    y_test = [clf_fold.get_category_index(y) for y in y_data[test_ix]]
                else:
                    y_test = [[clf_fold.get_category_index(y) for y in yy]
                              for yy in y_data[test_ix]]
                x_test = x_data[test_ix]

                Evaluation.__grid_search_loop__(
                    clf_fold, x_test, y_test, s, l, p, a, k_fold, i_fold,
                    def_cat, tag, categories, cache,
                    leave_pbar=False, extended_pbar=extended_pbar,
                    desc_pbar="[fold %d/%d] Grid search" % (i_fold + 1, k_fold), prep=prep
                )
                Evaluation.__cache_update__()

                x_test, y_test = None, None  # free memory

            Print.verbosity_region_end()

        return Evaluation.get_best_hyperparameters(metric, metric_target)


class Dataset:
    """A helper class with methods to read datasets from disk."""

    @staticmethod
    def load_from_files(data_path, folder_label=True, as_single_doc=False, sep_doc='\n'):
        r"""
        Load training/test documents and category labels from disk.

        :param data_path: the training or the test set path
        :type data_path: str
        :param folder_label: if True, read category labels from folders,
                             otherwise, read category labels from file names.
                             (default: True)
        :type folder_label: bool
        :param as_single_doc: read the documents as a single (and big) document
                              (default: False)
        :type as_single_doc: bool
        :param sep_doc: the separator/delimiter used to separate each document
                        when loading training/test documents from single file. Valid
                        only when ``folder_label=False``. (default: ``'\n'``)
        :type sep_doc: str
        :returns: the (x_train, y_train) or (x_test, y_test) pairs.
        :rtype: tuple
        """
        x_data = []
        y_data = []
        cat_info = {}

        Print.info("reading files...")

        if not folder_label:

            files = listdir(data_path)
            progress_bar = tqdm(
                total=len(files), desc="Loading documents",
                disable=Print.is_quiet()
            )
            for file in files:
                file_path = path.join(data_path, file)
                if path.isfile(file_path):
                    cat = path.splitext(file)[0]
                    progress_bar.set_description_str("Loading '%s' documents" % cat)

                    with open(file_path, "r", encoding=ENCODING) as fcat:
                        docs = (fcat.read().split(sep_doc)
                                if not as_single_doc else [fcat.read()])

                    x_data.extend(docs)
                    y_data.extend([cat] * len(docs))

                    cat_info[cat] = len(docs)
                progress_bar.update(1)
            progress_bar.close()
        else:
            folders = listdir(data_path)
            for icat, cat in enumerate(folders):
                cat_path = path.join(data_path, cat)
                if not path.isfile(cat_path):
                    cat_info[cat] = 0
                    files = listdir(cat_path)
                    pbar_desc = "[%d/%d] Loading '%s' documents" % (icat + 1, len(folders), cat)
                    for file in tqdm(files, desc=pbar_desc,
                                     leave=icat + 1 >= len(folders),
                                     disable=Print.is_quiet()):
                        file_path = path.join(cat_path, file)
                        if path.isfile(file_path):
                            with open(file_path, "r", encoding=ENCODING) as ffile:
                                x_data.append(ffile.read())
                                y_data.append(cat)
                            cat_info[cat] += 1

        Print.info("%d categories found" % len(cat_info))
        for cat in cat_info:
            Print.info(
                "'%s'%s"
                %
                (
                    cat,
                    '' if as_single_doc else " (%d documents)" % cat_info[cat]
                ),
                offset=4
            )

        return x_data, y_data

    @staticmethod
    def load_from_files_multilabel(docs_path, labels_path, sep_label=None, sep_doc='\n'):
        r"""
        Multilabel version of the ``Dataset.load_from_files()`` function.

        Load training/test documents and category labels from disk.

        :param docs_path: the file or the folder containing the training/test
                          documents.
        :type docs_path: str
        :param labels_path: the file containing the labels for each document.

                            * if ``docs_path`` is a file, then the ``labels_path`` file
                            should contain a line with the corresponding list of category
                            labels for each document in ``docs_path``. For
                            instance, if ``sep_doc='\n'`` and the the content
                            of ``docs_path`` is:

                            .. parsed-literal::
                                this is document 1
                                this is document 2
                                this is document 3

                            then, if ``sep_label=';'``, the ``labels_path``
                            file should contain the labels for each document
                            (in order) separated by ;, as follows:

                            .. parsed-literal::
                                labelA;labelB
                                labelA
                                labelB;labelC

                            * if ``docs_path`` is a folder containing the documents, then
                            the ``labels_path`` file should contain a line for each document and
                            category label. Each line should have the following format:
                            ``document_name<the sep_label>label``. For instance, if the
                            ``docs_path`` folder contains the following 3 documents:

                            .. parsed-literal::
                                doc1.txt
                                doc2.txt
                                doc3.txt

                            Then, following the above example, the ``labels_path`` file should be:

                            .. parsed-literal::
                                doc1    labelA
                                doc1    labelB
                                doc2    labelA
                                doc3    labelB
                                doc3    labelC

        :type labels_path: str
        :param sep_label: the separator/delimiter used to separate either each label (if
                          ``docs_path`` is a file) or the document name from its category
                          (if ``docs_path`` is a folder).
                          (default: ``';'`` when ``docs_path`` is a file, the ``'\s+'`` regular
                          expression otherwise).
        :type sep_label: str
        :param sep_doc: the separator/delimiter used to separate each document
                        when loading training/test documents from single file. Valid
                        only when ``folder_label=False``. (default: ``\n'``)
        :type sep_doc: str
        :returns: the (x_train, y_train) or (x_test, y_test) pairs.
        :rtype: tuple
        :raises: ValueError
        """
        x_data = []
        y_data = []
        cat_info = defaultdict(int)

        Print.info("reading files...")

        if path.isdir(docs_path):
            sep_label = sep_label or r'\s+'  # default separator

            with open(labels_path, "r", encoding=ENCODING) as flabels:
                doc_labels_raw = [re.split(sep_label, l.rstrip())
                                  for l in flabels.read().split('\n')]
                doc_labels = {}
                doc_names = []

                for i_doc_label, doc_label in enumerate(doc_labels_raw):
                    if len(doc_label) == 2:
                        doc_name, label = doc_label
                    elif len(doc_label) > 2:
                        doc_name, label = doc_label[0], " ".join(doc_label[1:])
                    else:
                        if doc_label[0] not in doc_labels:
                            doc_labels[doc_label[0]] = []
                        continue

                    if doc_name not in doc_labels:
                        doc_labels[doc_name] = [label]
                        doc_names.append(doc_name)
                    elif label:
                        doc_labels[doc_name].append(label)
                    cat_info[label] += 1

                for doc_name in tqdm(doc_names, desc="Loading documents"):
                    file_name = doc_name + ".txt" if '.' not in doc_name else doc_name
                    with open(path.join(docs_path, file_name), "r", encoding=ENCODING) as fdoc:
                        x_data.append(fdoc.read())
                    y_data.append(doc_labels[doc_name])

        else:
            sep_label = sep_label or ';'

            with open(labels_path, "r", encoding=ENCODING) as flabels:
                y_data = [re.split(sep_label, l) if l else []
                          for l in tqdm(flabels.read().split('\n'))]
            with open(docs_path, "r", encoding=ENCODING) as fdocs:
                x_data = fdocs.read().split(sep_doc)

            if len(x_data) != len(y_data):
                x_data = x_data[:len(y_data)]

            for labels in y_data:
                for i, label in enumerate(labels):
                    label = label.strip()
                    labels[i] = label
                    cat_info[label] += 1

        Print.info("%d categories found" % len(cat_info))
        for cat in cat_info:
            Print.info(
                "'%s'%s"
                %
                (cat, " (%d documents)" % cat_info[cat]),
                offset=4
            )

        return x_data, y_data

    # TODO: save_to_files(x_train, y_train, x_test, y_test)


class RecursiveDefaultDict(dict):
    """A dict whose default value is a dict."""

    def __missing__(self, key):
        """Class constructor."""
        value = self[key] = type(self)()
        return value


class Preproc:
    """A helper class with methods for pre-processing documents."""

    @staticmethod
    def clean_and_ready(text, dots=True, normalize=True, min_len=1):
        """Clean and prepare the text."""
        if normalize:
            try:
                text = text.decode("utf-8")
            except BaseException:
                pass

            text = ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if unicodedata.category(c) != 'Mn'
            ).replace("'", '')

        # tokenizing terms related to numbers
        text = REGEX_NUMBER.sub(
            "NNBRR",
            REGEX_PERCENT.sub(
                "NNBRRP",
                REGEX_MONEY.sub(
                    "MNNBRR",
                    REGEX_DATE.sub(
                        "NNBRRD",
                        REGEX_TEMP.sub(
                            "NNBRRT",
                            text + " "
                        )
                    )
                )
            )
        )

        # cleaning up the text
        if not dots:
            return (
                " ".join(re.findall("[a-zA-Z\n]{%d,}" % min_len, text))
            ).lower()
        else:
            text = REGEX_DOTS_CHARS.sub(
                ".",
                REGEX_DOTS_CHAINED.sub(r"\1.", text)
            ) + "."
            return (
                " ".join(re.findall(r"[a-zA-Z\n]{%d,}|\." % min_len, text))
            ).lower()


class Style:
    """Helper class to handle print styles."""

    __header__ = '\033[95m'
    __okblue__ = '\033[94m'
    __okgreen__ = '\033[92m'
    __warning__ = '\033[93m'
    __fail__ = '\033[91m'
    __endc__ = '\033[0m'
    __bold__ = '\033[1m'
    __underline__ = '\033[4m'

    @staticmethod
    def __apply__(text, format_str):
        """Appply a given style."""
        return "%s%s%s" % (format_str, text, Style.__endc__)

    @staticmethod
    def bold(text):
        """Apply bold style to ``text``."""
        return Style.__apply__(text, Style.__bold__)

    @staticmethod
    def ubold(text):
        """Apply underline and bold style to ``text``."""
        return Style.underline((Style.bold(text)))

    @staticmethod
    def underline(text):
        """Underline ``text``."""
        return Style.__apply__(text, Style.__underline__)

    @staticmethod
    def fail(text):
        """Apply the 'fail' style to ``text``."""
        return Style.__apply__(text, Style.__fail__)

    @staticmethod
    def warning(text):
        """Apply the 'warning' style to ``text``."""
        return Style.__apply__(text, Style.__warning__)

    @staticmethod
    def green(text):
        """Apply 'green' style to ``text``."""
        return Style.__apply__(text, Style.__okgreen__)

    @staticmethod
    def blue(text):
        """Apply 'blue' style to ``text``."""
        return Style.__apply__(text, Style.__okblue__)

    @staticmethod
    def header(text):
        """Apply 'header' style to ``text``."""
        return Style.__apply__(text, Style.__header__)


class Print:
    """Helper class to handle print functionalities."""

    __error_start__ = '*** '
    __error_end__ = ''
    __warn_start__ = '* '
    __warn_end__ = ''
    __info_start__ = '[ '
    __info_end__ = ' ]'

    style = Style

    __verbosity__ = VERBOSITY.NORMAL
    __verbosity_region_stack__ = []

    @staticmethod
    def error(msg='', raises=None, offset=0, decorator=True):
        """
        Print an error.

        :param msg: the message to show
        :type msg: str
        :param raises: the exception to be raised after showing the message
        :type raises: Exception
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use error message decoretor
        :type decorator: bool
        """
        print(
            Style.fail("%s%s%s%s" % (
                " " * offset,
                Print.__error_start__ if decorator else '',
                str(msg),
                Print.__error_end__ if decorator else ''
            ))
        )
        if raises:
            raise raises

    @staticmethod
    def warn(msg='', newln=True, raises=None, offset=0, decorator=True):
        """
        Print a warning message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param raises: the exception to be raised after showing the message
        :type raises: Exception
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use warning message decoretor
        :type decorator: bool
        """
        Print.show(
            Style.warning("%s%s%s%s" % (
                " " * offset,
                Print.__warn_start__ if decorator else '',
                str(msg),
                Print.__warn_end__ if decorator else '',
            )), newln
        )
        if raises:
            raise raises

    @staticmethod
    def info(msg='', newln=True, offset=0, decorator=True, force_show=False):
        """
        Print an info message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use info message decoretor
        :type decorator: bool
        :param force_show: if True, show message even when not in verbose mode
        :type force_show: bool
        """
        if Print.is_verbose() or force_show:
            Print.show(
                Style.blue("%s%s%s%s" % (
                    " " * offset,
                    Print.__info_start__ if decorator else '',
                    str(msg),
                    Print.__info_end__ if decorator else '',
                )), newln
            )

    @staticmethod
    def show(msg='', newln=True, offset=0, force_show=False):
        """
        Print a message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        """
        if Print.is_verbose() or force_show:
            print((" " * offset) + str(msg), end='\n' if newln else '')

    @staticmethod
    def set_decorator_info(start, end=None):
        """
        Set info messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__info_start__ = start or Print.__info_start__
        Print.__info_end__ = end or Print.__info_end__

    @staticmethod
    def set_decorator_warn(start, end=None):
        """
        Set warning messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__warn_start__ = start or Print.__warn_start__
        Print.__warn_end__ = end or Print.__warn_end__

    @staticmethod
    def set_decorator_error(start, end=None):
        """
        Set error messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__error_start__ = start or Print.__error_start__
        Print.__error_end__ = end or Print.__error_end__

    @staticmethod
    def set_verbosity(level):
        """
        Set the verbosity level.

            - ``0`` (quiet): do not output any message (only error messages)
            - ``1`` (normal): default behavior, display only warning messages and progress bars
            - ``2`` (verbose): display also the informative non-essential messages

        The following built-in constants can also be used to refer to these 3 values:
        ``VERBOSITY.QUIET``, ``VERBOSITY.NORMAL``, and ``VERBOSITY.VERBOSE``, respectively.

        For example, if you want PySS3 to hide everything, even progress bars, you could do:

        >>> from pyss3.util import Print, VERBOSITY
        ...
        >>> Print.set_verbosity(VERBOSITY.QUIET)  # or, equivalently, Print.set_verbosity(0)
        ...
        >>> # here's the rest of your code :D

        :param level: the verbosity level
        :type level: int
        """
        Print.__verbosity__ = level

    @staticmethod
    def get_verbosity():
        """
        Return the verbosity level.

            - ``0`` (quiet): do not output any message (only error messages)
            - ``1`` (normal): default behavior, display only warning messages and progress bars
            - ``2`` (verbose): display also the informative non-essential messages

        :returns: the verbosity level
        :rtype: int
        """
        return Print.__verbosity__

    @staticmethod
    def is_quiet():
        """Check if the current verbosity level is quiet."""
        return Print.__verbosity__ == VERBOSITY.QUIET

    @staticmethod
    def is_verbose():
        """Check if the current verbosity level is verbose."""
        return Print.__verbosity__ >= VERBOSITY.VERBOSE

    @staticmethod
    def verbosity_region_begin(level, force=False):
        """
        Indicate that a region with different verbosity begins.

        When the region ends by calling ``verbosity_region_end``, the previous
        verbosity will be restored.

        Example:

        >>> from pyss3.util import Print,VERBOSITY
        ...
        >>> Print.verbosity_region_begin(VERBOSITY.QUIET)
        >>> # inside this region (from now on), verbosity will be 'quiet'
        ...
        >>> Print.verbosity_region_end()
        >>> # the verbosity level is restored to what it was before entering the region

        :param level: the verbosity level for this region
                      (see ``set_verbosity`` documentation for valid values)
        :type level: int
        """
        Print.__verbosity_region_stack__.append(Print.__verbosity__)
        if force or not Print.is_quiet():
            Print.__verbosity__ = level

    @staticmethod
    def verbosity_region_end():
        """
        Indicate that a region with different verbosity ends.

        The verbosity will be restored to the value it had
        before beginning this region with ``verbosity_region_begin``.

        Example:

        >>> from pyss3.util import Print,VERBOSITY
        ...
        >>> Print.verbosity_region_begin(VERBOSITY.VERBOSE)
        >>> # inside this region (from now on), verbosity will be 'verbose'
        ...
        >>> Print.verbosity_region_end()
        >>> # the verbosity level is restored to what it was before entering the region
        """
        Print.__verbosity__ = Print.__verbosity_region_stack__.pop()


def round_fix(v, precision=4):
    """Round the number v (used to keep the results history file small)."""
    try:
        return round(float(v), precision)
    except (ValueError, TypeError, AttributeError):
        raise TypeError(ERROR_IHV)


def list_by_force(v):
    """Convert any non-iterable object into a list."""
    try:
        iter(v)
        return v
    except TypeError:
        return [v]


def is_a_collection(o):
    """Return True when the object ``o`` is a collection."""
    return hasattr(o, "__getitem__") and ((PY2 and not isinstance(o, basestring)) or
                                          (not PY2 and not isinstance(o, (str, bytes))))


def membership_matrix(clf, y_data, labels=True, show_pbar=True):
    """
    Transform a list of (multiple) labels into a "membership matrix".

    The membership matrix consists converting each list of category labels
    (i.e., each *y* in ``y_data``) into a vector in which there's a fixed
    position associated to each learned category, having the value 1 for each
    label in *y*, and 0 otherwise.

    When working with multi-label classification problems, this representation
    enables measuring the performance using common evaluation metrics such
    as Hamming loss, exact match ratio, accuracy, precision, recall, F1, etc.

    For instance, suppose ``y_data = [[], ['labelA'], ['labelB'], ['labelA', 'labelC']]``
    and that the classifier ``clf`` has been trained on 3 categories whose labels are
    'labelA', 'labelB', and 'labelC', then, we would have that:

    >>> membership_matrix(clf, [[], ['labelA'], ['labelB'], ['labelA', 'labelC']])

    returns the following membership matrix:

    >>> [[0, 0, 0],  # []
    >>>  [1, 0, 0],  # ['labelA']
    >>>  [0, 1, 0],  # ['labelB']
    >>>  [1, 0, 1]]  # ['labelA', 'labelC']

    :param clf: the trained classifier
    :type clf: SS3
    :param y_data: the list of document labels
    :type y_data: list of list of str
    :param labels: whether the `y_data` list contains category labels or
                   category indexes (default: True)
    :type labels: bool
    :param show_pbar: whether to show the progress bar or not (default: True)
    :type show_pbar: bool
    :returns: a (sparse) matrix in which each row is the membership vector of
              each element (labels) in ``y_data``.
    :rtype: scipy.sparse.lil.lil_matrix
    :raises: ValueError
    """
    if not clf.__categories__:
        raise ValueError("The `clf` classifier has not been trained yet!")

    from scipy import sparse

    labels2index = dict([(c if labels else clf.get_category_index(c), i)
                         for i, c in enumerate(clf.get_categories())])
    y_data_matrix = sparse.lil_matrix((len(y_data), len(labels2index)), dtype="b")
    li = np.array([[i, labels2index[l]]
                   for i, ll in enumerate(y_data)
                   for l in ll
                   if l in labels2index])
    if len(li) > 0:
        y_data_matrix[li[:, 0], li[:, 1]] = 1

    return y_data_matrix
