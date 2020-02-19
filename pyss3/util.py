# -*- coding: utf-8 -*-
"""This is a helper module with utility classes and functions."""
from __future__ import print_function
from io import open
from os import listdir, path
from tqdm import tqdm

from math import ceil
from numpy import mean, linspace, arange
from itertools import product
from os import remove as remove_file
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import numpy as np
import unicodedata
import webbrowser
import json
import re

ENCODING = "utf-8"

RE_URL_NOISE = "(?P<url_noise>%s|%s|%s|%s)" % (
    r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}"
    r"\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"(?:\.[a-zA-Z]{2,5}\?[^\s]+ )",
    r"(?:/[a-zA-Z0-9_]+==)",
    r"(?:(([a-zA-Z]+://)?([a-zA-Z]+\.)?([a-zA-Z]{1,7}\.[a-zA-Z]{2,7})?"
    r"/[a-zA-Z0-9]*[A-Z0-9][a-zA-Z0-9]*(\?[a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)?)|"
    r"((\?[a-zA-Z0-9]+=[a-zA-Z0-9./:=?#_-]*)"
    r"(&[a-zA-Z0-9]+=[a-zA-Z0-9./:=?#_-]*)*) )"
)
REGEX_REMOVE_URL_NOISE = re.compile(RE_URL_NOISE)
REGEX_CAMELCASE = re.compile(r"#(?P<camel>[A-Z][a-z]+|[A-Z][A-Z]+)")
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

EVAL_HTML_OUT_FILE = "./ss3_model_evaluation[%s].html"
EVAL_HTML_SRC_FOLDER = "resources/model_evaluation/"
EVAL_CACHE_EXT = ".ss3ev"

STR_ACCURACY, STR_PRECISION = "accuracy", "precision"
STR_RECALL, STR_F1 = "recall", "f1-score"
METRICS = [STR_PRECISION, STR_RECALL, STR_F1]
EXCP_METRICS = [STR_ACCURACY, "confusion_matrix", "categories"]
AVGS = ["micro avg", "macro avg", "weighted avg"]

STR_TEST, STR_FOLD = 'test', 'fold'
STR_MOST_PROBABLE = "most-probable"

ERROR_MNE = "error: the classifier has not been evaluated yet"

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
    __cache__ = None
    __cache_file__ = None
    __clf__ = None
    __last_eval_tag__ = None
    __last_eval_method__ = None
    __last_eval_def_cat__ = None

    @staticmethod
    def __k_fold2method__(k_fold):
        """Convert the k number to a proper method string."""
        return STR_TEST if k_fold <= 1 else str(k_fold) + '-' + STR_FOLD

    @staticmethod
    def __set_last_evaluation__(tag, method, def_cat):
        Evaluation.__last_eval_tag__ = tag
        Evaluation.__last_eval_method__ = method
        Evaluation.__last_eval_def_cat__ = def_cat

    @staticmethod
    def __get_last_evaluation__():
        if Evaluation.__last_eval_tag__:
            return Evaluation.__last_eval_tag__, \
                Evaluation.__last_eval_method__, \
                Evaluation.__last_eval_def_cat__
        elif Evaluation.__cache__:
            tag = Evaluation.__cache__.keys()[0]
            method = Evaluation.__cache__[tag].keys()[0]
            def_cat = Evaluation.__cache__[tag][method].keys()[0]
            return tag, method, def_cat
        else:
            None, None, None

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
        """Save results history (evaluations) to disk."""
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
        cache, categories, accuracy, report, conf_matrix, k_fold, i_fold, s, l, p, a
    ):
        """Save evaluation results to disk."""
        rf = round_fix
        s, l = rf(s), rf(l)
        p, a = rf(p), rf(a)

        cache["categories"] = categories

        # if there aren't previous best results, initialize them to -1
        if cache["accuracy"]["best"]["value"] == {}:
            cache["accuracy"]["best"]["value"] = -1
            for metric, avg in product(METRICS, AVGS):
                if avg in report:  # scikit-learn > 0.20 does not include 'micro avg' in report
                    cache[metric][avg]["best"]["value"] = -1
            for cat in categories:
                for metric in METRICS:
                    cache[metric]["categories"][cat]["best"]["value"] = -1

        # if fold results array is empty, create new ones
        if cache["accuracy"]["fold_values"][s][l][p][a] == {}:
            cache["accuracy"]["fold_values"][s][l][p][a] = [0] * k_fold
            cache["confusion_matrix"][s][l][p][a] = [None] * k_fold
            for metric, avg in product(METRICS, AVGS):
                if avg in report:
                    cache[metric][avg]["fold_values"][s][l][p][a] = [0] * k_fold
            for cat in categories:
                for metric in METRICS:
                    cache[metric]["categories"][cat]["fold_values"][s][l][p][a] = [0] * k_fold

        # saving fold results
        cache["accuracy"]["fold_values"][s][l][p][a][i_fold] = rf(accuracy)
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
        """Return whether this evaluation is already computed."""
        s, l = round_fix(s), round_fix(l)
        p, a = round_fix(p), round_fix(a)
        results = Evaluation.__cache_get_evaluations__(tag, method, def_cat)
        return results["accuracy"]["value"][s][l][p][a] != {}

    @staticmethod
    def __cache_get_test_evaluation__(tag, def_cat, s, l, p, a):
        """Return test results from cache."""
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
        """Given a tag, a method and a default category return results history."""
        Evaluation.__cache_load__()
        return Evaluation.__cache__[tag][method][def_cat]

    @staticmethod
    def __cache_remove_lpsa__(c_metric, s, l, p, a, simulate=False, best=True):
        """Remove evaluations from history given hyperparameters s, l, p, a."""
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
    def __cache_get_default_tag__(clf, n_grams=None):
        n_grams = n_grams or len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1
        return "%s_%s" % (
            clf.get_name(),
            "words" if n_grams == 1 else "%d-grams" % n_grams
        )

    @staticmethod
    def __cache_remove__(tag, method, def_cat, s, l, p, a, simulate=False):
        """Remove evaluations from history."""
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
    def __classification_report_k_fold__(tag, method, def_cat, s, l, p, a, plot=True):
        """Create the classification report for k-fold validations."""
        verbosity = Print.get_verbosity()
        Print.set_verbosity(VERBOSITY.VERBOSE)

        s, l = round_fix(s), round_fix(l)
        p, a = round_fix(p), round_fix(a)

        cache = Evaluation.__cache_get_evaluations__(tag, method, def_cat)
        categories = cache["categories"]

        name_width = max(len(cn) for cn in categories)
        width = max(name_width, len(AVGS[-1]))
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(METRICS)

        report = head_fmt.format('', *['avg'] * len(METRICS), width=width)
        report += '\n'
        report += head_fmt.format('', *METRICS, width=width)
        report += '\n\n'

        for cat in categories:
            report += '{:>{width}s} '.format(cat, width=width)
            for metric in METRICS:
                report += ' {:>9.2f}'.format(
                    cache[metric]["categories"][cat]["value"][s][l][p][a]
                )
            report += '\n'
        report += '\n'
        for avg in AVGS:
            if avg in cache[metric]:
                report += '{:>{width}s} '.format(avg, width=width)
                for metric in METRICS:
                    report += ' {:>9.2f}'.format(
                        cache[metric][avg]["value"][s][l][p][a]
                    )
                report += '\n'

        report += "\n\n %s: %.3f\n" % (
            Print.style.bold("avg accuracy"), cache["accuracy"]["value"][s][l][p][a]
        )

        Print.show(report)
        Print.set_verbosity(verbosity)

        if plot:
            Evaluation.__plot_confusion_matrices__(
                cache["confusion_matrix"][s][l][p][a], categories,
                r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
                %
                (s, l, p, a)
            )

        return cache["accuracy"]["value"][s][l][p][a]

    @staticmethod
    def __plot_confusion_matrices__(cms, classes, info='', max_colums=3):
        """Show and plot the confusion matrices."""
        import matplotlib.pyplot as plt

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

            if (axi % max_colums) == 0:
                ax.set_ylabel('True', fontweight="bold")
                ax.set_yticklabels(classes)
            else:
                ax.tick_params(labelleft=False)

            if axi + 1 > n_cms - max_colums:
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
        tag=None, folder=False, plot=True, k_fold=1, i_fold=0, force_show=False
    ):
        """Compute evaluation results and save them to disk."""
        import warnings
        from . import STR_UNKNOWN_CATEGORY, IDX_UNKNOWN_CATEGORY, STR_UNKNOWN
        warnings.filterwarnings('ignore')

        verbosity = Print.get_verbosity()
        if force_show:
            Print.set_verbosity(VERBOSITY.VERBOSE)

        n_cats = len(categories)
        if def_cat == STR_UNKNOWN:
            if categories[-1] != STR_UNKNOWN_CATEGORY:
                categories += [STR_UNKNOWN_CATEGORY]
            y_pred[:] = [y if y != IDX_UNKNOWN_CATEGORY else n_cats for y in y_pred]

        accuracy = accuracy_score(y_pred, y_true)
        Print.show()
        Print.show(
            classification_report(
                y_true, y_pred,
                labels=range(len(categories)), target_names=categories
            )
        )
        Print.show(
            "\n %s: %.3f"
            %
            (Print.style.bold("accuracy"), accuracy)
        )

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

        conf_matrix = confusion_matrix(y_true, y_pred)
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
                s, l, p, a
            )

        if plot:
            Evaluation.__plot_confusion_matrices__(
                [conf_matrix], categories,
                r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
                %
                (s, l, p, a)
            )

        Print.set_verbosity(verbosity)
        warnings.filterwarnings('default')

        return accuracy

    @staticmethod
    def __grid_search_loop__(
        clf, x_test, y_test, ss, ll, pp, aa, k_fold,
        i_fold, def_cat, tag, categories, cache=True, leave_pbar=True
    ):
        """Grid search main loop."""
        ss = [round_fix(s) for s in ss]
        ll = [round_fix(l) for l in ll]
        pp = [round_fix(p) for p in pp]
        aa = [round_fix(a) for a in aa]

        slp_list = list(product(ss, ll, pp))
        progress_bar = tqdm(
            total=len(slp_list) * len(aa),
            desc=" Grid Search", leave=leave_pbar
        )
        progress_desc = tqdm(
            total=0,
            bar_format='{desc}', leave=leave_pbar
        )

        method = Evaluation.__k_fold2method__(k_fold)

        verbosity = Print.get_verbosity()
        Print.set_verbosity(VERBOSITY.QUIET)

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
                        x_test, def_cat, labels=False, leave_pbar=False
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

        Print.set_verbosity(verbosity)

    @staticmethod
    def set_classifier(clf):
        if Evaluation.__clf__ != clf:
            Evaluation.__clf__ = clf
            Evaluation.__cache_file__ = path.join(
                clf.__models_folder__,
                clf.get_name() + EVAL_CACHE_EXT
            )
            Evaluation.__cache__ = None

    @staticmethod
    def clear_cache():
        Evaluation.__cache__ = None
        clf = Evaluation.__clf__
        if clf:
            if path.exists(Evaluation.__cache_file__):
                remove_file(Evaluation.__cache_file__)

    @staticmethod
    def plot(open_browser=True):
        """Save results history (evaluations) to disk (interactive html file)."""
        clf = Evaluation.__clf__
        Evaluation.__cache_load__()

        if not Evaluation.__cache__:
            Print.warn("no evaluations to be plotted")
            return False

        pyss3_path = path.dirname(__file__)
        html_src = EVAL_HTML_SRC_FOLDER
        result_html_file = EVAL_HTML_OUT_FILE % clf.__name__
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
            else:
                fout.write(line)

        fhtml.close()
        fout.close()

        if open_browser:
            webbrowser.open(result_html_file)

        Print.info("evaluation plot saved in '%s'" % result_html_file)
        return True

    @staticmethod
    def get_best_hyperparameters(metric='accuracy', avg='macro avg',
                                 tag=None, method=None, def_cat=None):
        if metric != STR_ACCURACY and metric not in METRICS:
            raise ValueError(
                "`metric` argument: '%s' is not a valid metric (excepted: %s)"
                %
                (str(metric), ", ".join(["'%s'" % m for m in [STR_ACCURACY] + METRICS]))
            )

        if avg not in AVGS:
            raise ValueError(
                "`avg` argument: '%s' is not a valid average (excepted: %s)"
                %
                (str(metric), ", ".join(["'%s'" % a for a in AVGS]))
            )

        l_tag, l_method, l_def_cat = Evaluation.__get_last_evaluation__()
        tag, method, def_cat = tag or l_tag, method or l_method, def_cat or l_def_cat

        c_metric = Evaluation.__cache__[tag][method][def_cat][metric]

        if metric == STR_ACCURACY:
            best = c_metric["best"]
        else:
            best = c_metric[avg]["best"]

        if not best:
            raise ValueError(ERROR_MNE)

        return best["s"], best["l"], best["p"], best["a"]

    @staticmethod
    def show_best(tag=None, method=None, def_cat=None, metric=None, avg=None):
        """Print evaluations best values."""
        Evaluation.__cache_load__()
        cache = Evaluation.__cache__
        ps = Print.style

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

                    evl = cache[t][md][dc]["accuracy"]["value"]
                    n_evl = len([
                        a for s in evl for l in evl[s]
                        for p in evl[s][l] for a in evl[s][l][p]
                    ])
                    print("(%d evaluations)" % n_evl)

                    best = cache[t][md][dc]["accuracy"]["best"]
                    print(
                        "    Best %s: %s %s" % (
                            ps.green("accuracy"), ps.warning(best["value"]),
                            ps.blue("(s %s l %s p %s a %s)") % (
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
                                    ps.blue("(s %s l %s p %s a %s)") % (
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
                                        ps.blue("(s %s l %s p %s a %s)")
                                        % (
                                            best["s"], best["l"],
                                            best["p"], best["a"]
                                        )
                                    ))

        print()

    @staticmethod
    def remove(s=None, l=None, p=None, a=None, method=None, def_cat=None, tag=None, simulate=False):
        """Evaluation remove command handler."""
        clf = Evaluation.__clf__
        count, count_details = None, None

        if not clf:
            return count, count_details

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
    def test(clf, x_test, y_test, def_cat=STR_MOST_PROBABLE, tag=None, plot=True, cache=True):
        """Test the model with a given test set."""
        Evaluation.set_classifier(clf)
        tag = tag or Evaluation.__cache_get_default_tag__(clf)
        s, l, p, a = clf.get_hyperparameters()

        Evaluation.__set_last_evaluation__(tag, STR_TEST, def_cat)

        y_pred = None
        if cache:
            _y_test, y_pred, categories = Evaluation.__cache_get_test_evaluation__(
                tag, def_cat, s, l, p, a
            )

        # if not cached
        if not y_pred:
            clf.set_hyperparameters(s, l, p, a)
            y_pred = clf.predict(x_test, def_cat, labels=False)
            categories = clf.get_categories()
            y_test[:] = [clf.get_category_index(y) for y in y_test]
        else:
            y_test = _y_test

        return Evaluation.__evaluation_result__(
            clf, y_test, y_pred, categories,
            def_cat, cache, STR_TEST,
            tag, plot=plot, force_show=True
        )

    @staticmethod
    def k_fold(
        clf, x_data, y_data, k_fold=4, n_grams=None,
        def_cat=STR_MOST_PROBABLE, tag=None, plot=True, cache=True
    ):
        """
        Perform a stratified k-fold cross validation using the given data.

        :raises: InvalidCategoryError
        """
        from . import SS3
        Evaluation.set_classifier(clf)
        n_grams = n_grams or (len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1)
        tag = tag or Evaluation.__cache_get_default_tag__(clf, n_grams)

        Print.verbosity_region_begin(VERBOSITY.NORMAL)
        method = Evaluation.__k_fold2method__(k_fold)

        Evaluation.__set_last_evaluation__(tag, method, def_cat)

        s, l, p, a = clf.get_hyperparameters()
        categories = clf.get_categories()
        x_data, y_data = np.array(x_data), np.array(y_data)
        skf = StratifiedKFold(n_splits=k_fold)
        progress_bar = tqdm(total=k_fold, desc=" K-Fold Progress")
        for i_fold, (train_ix, test_ix) in enumerate(skf.split(x_data, y_data)):
            if not cache or not Evaluation.__cache_is_in__(
                tag, method, def_cat, s, l, p, a
            ):
                x_train, y_train = x_data[train_ix], y_data[train_ix]
                y_test = [clf.get_category_index(y) for y in y_data[test_ix]]
                x_test = x_data[test_ix]

                _clf = SS3()
                _clf.set_hyperparameters(s, l, p, a)
                _clf.fit(x_train, y_train, n_grams, leave_pbar=False)

                y_pred = _clf.predict(
                    x_test, def_cat, labels=False, leave_pbar=False
                )

                Evaluation.__evaluation_result__(
                    _clf, y_test, y_pred,
                    categories, def_cat,
                    cache, method, tag,
                    plot=False, k_fold=k_fold, i_fold=i_fold
                )

            progress_bar.update(1)

        progress_bar.close()
        Print.verbosity_region_end()

        Print.show()
        return Evaluation.__classification_report_k_fold__(
            tag, method, def_cat, s, l, p, a, plot=plot
        )

    @staticmethod
    def grid_search(
        clf, x_data, y_data, s=None, l=None, p=None, a=None,
        k_fold=None, n_grams=None, def_cat=STR_MOST_PROBABLE, tag=None,
        metric='accuracy', avg='macro avg', cache=True
    ):
        """
        Perform a grid search using values from `s`, ``l``, ``p``, ``a``.

        :raises: InvalidCategoryError
        """
        from . import SS3
        Evaluation.set_classifier(clf)
        n_grams = n_grams or (len(clf.__max_fr__[0]) if len(clf.__max_fr__) > 0 else 1)
        tag = tag or Evaluation.__cache_get_default_tag__(clf, n_grams)
        method = Evaluation.__k_fold2method__(k_fold)

        Evaluation.__set_last_evaluation__(tag, method, def_cat)

        s = s or clf.get_s()
        l = l or clf.get_l()
        p = p or clf.get_p()
        a = a or clf.get_a()

        Print.show()
        if not k_fold:  # if test
            x_test, y_test = x_data, [clf.get_category_index(y) for y in y_data]
            Evaluation.__grid_search_loop__(
                clf, x_test, y_test, s, l, p, a, 1, 0,
                def_cat, tag, clf.get_categories(), cache
            )
        else:  # if k-fold
            Print.verbosity_region_begin(VERBOSITY.NORMAL)

            x_data, y_data = np.array(x_data), np.array(y_data)
            skf = StratifiedKFold(n_splits=k_fold)
            progress_bar = tqdm(
                position=0, total=k_fold,
                desc=" K-Fold Progress"
            )

            for i_fold, (train_ix, test_ix) in enumerate(
                skf.split(x_data, y_data)
            ):
                x_train, y_train = x_data[train_ix], y_data[train_ix]
                y_test = [clf.get_category_index(y) for y in y_data[test_ix]]
                x_test = x_data[test_ix]
                categories = clf.get_categories()

                _clf = SS3()
                _clf.fit(x_train, y_train, n_grams, leave_pbar=False)

                Evaluation.__grid_search_loop__(
                    _clf, x_test, y_test, s, l, p, a, k_fold, i_fold,
                    def_cat, tag, categories, cache, leave_pbar=False
                )

                Evaluation.__cache_update__()

                progress_bar.update(1)

            progress_bar.close()
            Print.verbosity_region_end()

        return Evaluation.get_best_hyperparameters(metric, avg)


class Dataset:
    """A helper class with methods to read/write datasets."""

    @staticmethod
    def load_from_files(data_path, folder_label=True, as_single_doc=False):
        """
        Load category documents from disk.

        :param data_path: the training or the test set path
        :type data_path: str
        :param folder_label: if True, read category labels from folders,
                             otherwise, read category labels from file names.
                             (default: True)
        :type folder_label: bool
        :param as_single_doc: read the documents as a single (and big) document
                              (default: False)
        :type folder_label: bool
        :returns: the (x_train, y_train) or the (x_test, y_test) pairs.
        :rtype: tuple
        """
        x_data = []
        y_data = []
        cat_info = {}

        Print.info("reading files...")

        if not folder_label:

            files = listdir(data_path)
            for file in tqdm(files, desc=" Category files",
                             leave=False, disable=Print.is_quiet()):
                file_path = path.join(data_path, file)
                if path.isfile(file_path):
                    cat = path.splitext(file)[0]

                    with open(file_path, "r", encoding=ENCODING) as fcat:
                        docs = (fcat.readlines()
                                if not as_single_doc else [fcat.read()])

                    x_data.extend(docs)
                    y_data.extend([cat] * len(docs))

                    cat_info[cat] = len(docs)
        else:

            folders = listdir(data_path)
            for item in tqdm(folders, desc=" Categories",
                             leave=False, disable=Print.is_quiet()):
                item_path = path.join(data_path, item)
                if not path.isfile(item_path):
                    cat_info[item] = 0
                    files = listdir(item_path)
                    for file in tqdm(files, desc=" Documents",
                                     leave=False, disable=Print.is_quiet()):
                        file_path = path.join(item_path, file)
                        if path.isfile(file_path):
                            with open(file_path, "r", encoding=ENCODING) as ffile:
                                x_data.append(ffile.read())
                                y_data.append(item)
                            cat_info[item] += 1

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

        # removing tweets links, noisy parameters in urls, etc.
        text = REGEX_REMOVE_URL_NOISE.sub(".", text)

        # resolving camel-cased words (e.g. #ThisTypeOfHashTags)
        text = REGEX_CAMELCASE.sub(r" \1 ", text)

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
    __verbosity_old__ = None

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
    def verbosity_region_begin(level):
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
        if not Print.is_quiet():
            Print.__verbosity_old__ = Print.__verbosity__
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
        if Print.__verbosity_old__ is not None:
            Print.__verbosity__ = Print.__verbosity_old__
            Print.__verbosity_old__ = None


def round_fix(v, precision=4):
    """Round the number v (used to keep the results history file small)."""
    return round(float(v), precision)
