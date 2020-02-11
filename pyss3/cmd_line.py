# -*- coding: utf-8 -*-
"""
This module lets you interact with your SS3 models through a Command Line.

(Please, visit https://github.com/sergioburdisso/pyss3 for more info)
"""
from __future__ import print_function
from os import listdir, path, remove as remove_file, rename as rename_file
from io import open
from cmd import Cmd
from tqdm import tqdm
from math import ceil
from numpy import mean, linspace
from itertools import product
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from .server import Server
from .util import Print, VERBOSITY, Dataset, RecursiveDefaultDict
from . import \
    SS3, InvalidCategoryError, STR_MODEL_EXT, \
    STR_UNKNOWN_CATEGORY, IDX_UNKNOWN_CATEGORY, __version__

import numpy as np
import webbrowser
import json
import sys
import re
try:
    import readline
except ImportError:
    readline = None

ENCODING = "utf-8"

# HISTFILE = path.expanduser('~/.ss3_history')
HISTFILE = '.ss3_history'
HISTFILE_SIZE = 1000

STOPWORDS_FILE = "./ss3_stopwords[%s].txt"
RESULT_HTML_OUT_FILE = "./ss3_model_evaluation[%s].html"
RESULT_HTML_SRC_FOLDER = "resources/model_evaluation/"
RESULT_HISTORY_EXT = ".ss3ev"

METRICS = ["precision", "recall", "f1-score"]
EXCP_METRICS = ["accuracy", "confusion_matrix", "categories"]
AVGS = ["micro avg", "macro avg", "weighted avg"]

ERROR_MR = "A model is required: either load a model or create a new one"
ERROR_AR = "Empty arguments: at least one argument must be given"
ERROR_WAN = "Wrong number of arguments: expected %d but received %d"
ERROR_WAT = "Wrong value type: excepted %s"
ERROR_UA = "Unknown argument: %s"
ERROR_ICN = "Invalid category name: %s"
ERROR_HIV = ("Invalid hyperparameter value: "
             "expected a float number but received '%s'")
ERROR_HVM = ("Hyperparameter value missing: "
             "the value for the hyperparameter '%s' is missing")
ERROR_NSF = "No such file: %s"
ERROR_NSD = "No such directory: %s"
ERROR_MNT = "Test not allowed: The model hasn't been trained yet"
ERROR_WNAUA = "Wrong number of arguments: there are unknown arguments"
ERROR_WNGRAM = ("Wrong n-grams argument value: "
                "N should be a positive integer (e.g. 2-grams, 3-grams, etc.)")
ERROR_WKFOLD = ("Wrong k-fold argument value: "
                "k should be an integer > 1 (e.g. 5-fold, 10-fold, etc.)")

MSG_USER_INPUT_DOC = ("Write the document below, to finish press Ctrl+D "
                      "(or Ctrl+Z for Windows users)")
WARN_OVERWRITE = ("This model already exists, do you really want to "
                  "overwrite it? [Y/n] ")
WARN_NO_STOPWORDS = ("There are no stopwords!\n"
                     "Suggestion: try with another threshold value or "
                     "improving your\n"
                     "*             model by changing some of its "
                     "hyperparameters")

STR_MODEL, STR_VOCABULARY, STR_STOPWORDS = "model", "vocabulary", "stopwords"
STR_PARAMETERS, STR_CATEGORIES, STR_ALL = "parameters", "categories", "all"
STR_INFO, STR_PLOT, STR_SAVE, STR_REMOVE = "info", "plot", "save", "remove"
STR_DISTRIBUTION, STR_EVALUATIONS = "distribution", "evaluations"
STR_FILE, STR_FOLDER, STR_VERBOSE = "file", "folder", "verbose"
STR_UNKNOWN, STR_MOST_PROBABLE = "unknown", "most-probable"
STR_FOLD, STR_TEST, STR_NGRAMS = "fold", "test", "grams"
STR_S, STR_L, STR_P, STR_A = "s", "l", "p", "a"
STR_NO_CACHE = "no-cache"

ARGS_HYP = [STR_S, STR_L, STR_P, STR_A]
ARGS_CATS = []  # the list of model's categories
ARGS = {
    "save": [STR_MODEL, STR_VOCABULARY, STR_STOPWORDS, STR_EVALUATIONS],
    "info": [STR_PARAMETERS, STR_CATEGORIES, STR_ALL, STR_EVALUATIONS],
    "set": ARGS_HYP,
    "learn": [STR_NGRAMS],
    "train": [STR_FILE, STR_FOLDER, STR_NGRAMS],
    "test": [STR_UNKNOWN, STR_MOST_PROBABLE, STR_NO_CACHE],
    "plot": [STR_DISTRIBUTION, STR_EVALUATIONS],
    "live_test": [STR_FILE, STR_FOLDER, STR_VERBOSE],
    "evaluations": ARGS_HYP + [
        STR_PLOT, STR_SAVE, STR_REMOVE, STR_INFO,
        STR_MOST_PROBABLE, STR_UNKNOWN, STR_TEST, STR_FOLD
    ],
    "grid_search": [STR_FOLD],
    "load": [],   # loaded later using models in './ss3_ss3_models'
}
ARGS["grid_search"] += ARGS["train"] + ARGS["test"] + ARGS_HYP

MODELS = []
RESULTS_HISTORY = None
CLF = None
F_PRECISION = 4

frange = linspace  # an alias for grid_search command
r = frange  # an alias for grid_search command


try:
    input = raw_input  # Python 2
except NameError:
    pass


class GetTestDataError(Exception):
    """Exception thrown when an error occur while retrieving the test data."""

    def __init__(self):
        """Class constructor."""
        pass


class ArgsParseError(Exception):
    """Exception thrown when an error occur parsing commands arguments."""

    def __init__(self):
        """Class constructor."""
        pass


def requires_model(func):
    """A @decorator."""
    def model_check(*args, **kwargs):
        if not CLF:
            Print.error(ERROR_MR)
            Print.warn(
                "Suggestion: use one of the commands 'load' or 'new'"
            )
            return None

        return func(*args, **kwargs)
    model_check.__name__ = func.__name__
    model_check.__doc__ = func.__doc__
    return model_check


def requires_args(func):
    """A @decorator."""
    def arg_check(*args, **kwargs):
        if not args[1]:
            Print.error(ERROR_AR)
            Print.warn(
                "Suggestion: use the command 'help %s' for more details"
                %
                func.__name__[3:]
            )
            return None

        return func(*args, **kwargs)
    arg_check.__name__ = func.__name__
    arg_check.__doc__ = func.__doc__
    return arg_check


def json2rh(dct):
    """Convert a given dictionary to a RecursiveDefaultDict."""
    r_ddct = RecursiveDefaultDict()
    for key in dct.keys():
        try:
            r_ddct[float(key)] = dct[key]
        except ValueError:
            r_ddct[key] = dct[key]
    return r_ddct


def split_args(args):
    """Parse and split arguments."""
    return [a.strip('"-') for a in re.findall(r'[^\s\"]+|".+"', args)]


def parse_hparams_args(op_args, defaults=True):
    """Parse hyperparameters arguments list."""
    used_args_ix = []
    hparams = {}

    if defaults:
        s, l, p, a = CLF.get_hyperparameters()
    else:
        s, l, p, a = None, None, None, None

    keys_args = ((STR_S, s), (STR_L, l), (STR_P, p), (STR_A, a))

    for key_args in keys_args:
        hp_str, h_v = key_args
        arg = intersect(key_args, op_args)
        if arg:
            argi = op_args.index(arg[0])
            used_args_ix.extend([argi, argi + 1])
            try:
                hparams[hp_str] = round_fix(op_args[argi + 1])
            except IndexError:
                Print.error(ERROR_HVM % hp_str, raises=ArgsParseError)
            except BaseException:
                Print.error(
                    ERROR_HIV % op_args[argi + 1], raises=ArgsParseError
                )
        else:
            hparams[hp_str] = h_v

    return hparams, used_args_ix


def k_fold2method(k_fold):
    """Convert the k number to a proper method string."""
    return STR_TEST if k_fold == 1 else str(k_fold) + '-' + STR_FOLD


def intersect(l0, l1):
    """Given two lists return the intersection."""
    return [e for e in l0 if e in l1]


def subtract(l0, l1):
    """Subtract list l1 from  l0."""
    return [e for e in l0 if e not in l1]


def re_in(regex, l):
    """Given a list of strings, return the first match in the list."""
    for e in l:
        match = re.match(regex, e)
        if match:
            return match
    return None


def module_path(file_path):
    """Convert a file path relative to this module path."""
    return path.join(path.dirname(__file__), file_path)


def round_fix(v):
    """Round the number v (used to keep the results history file small)."""
    return round(float(v), F_PRECISION)


def k_fold_classification_report(data_path, method, def_cat, s, l, p, a):
    """Create the classification report for k-fold validations."""
    s, l = round_fix(s), round_fix(l)
    p, a = round_fix(p), round_fix(a)

    rh = get_results_history(data_path, method, def_cat)
    categories = rh["categories"]

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
                rh[metric]["categories"][cat]["value"][s][l][p][a]
            )
        report += '\n'
    report += '\n'
    for avg in AVGS:
        if avg in rh[metric]:
            report += '{:>{width}s} '.format(avg, width=width)
            for metric in METRICS:
                report += ' {:>9.2f}'.format(
                    rh[metric][avg]["value"][s][l][p][a]
                )
            report += '\n'

    report += "\n\n %s: %.3f\n" % (
        Print.style.bold("avg accuracy"), rh["accuracy"]["value"][s][l][p][a]
    )

    Print.show(report)

    plot_confusion_matrices(
        rh["confusion_matrix"][s][l][p][a], categories,
        r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
        %
        (s, l, p, a)
    )


def plot_confusion_matrices(cms, classes, info='', max_colums=3):
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


def load_results_history():
    """Load results history (evaluations) from disk."""
    global RESULTS_HISTORY

    if not RESULTS_HISTORY:
        Print.info("loading results history")
        try:
            rh_path = path.join(
                CLF.__models_folder__,
                CLF.__name__ + RESULT_HISTORY_EXT
            )
            with open(rh_path, "r", encoding=ENCODING) as json_file:
                RESULTS_HISTORY = json.loads(
                    json_file.read(),
                    object_hook=json2rh
                )
        except IOError:
            Print.info("no results history found, creating a new one")
            RESULTS_HISTORY = RecursiveDefaultDict()


def save_results_history():
    """Save results history (evaluations) to disk."""
    Print.info("saving results")
    rh_path = path.join(
        CLF.__models_folder__,
        CLF.__name__ + RESULT_HISTORY_EXT
    )
    with open(rh_path, "w", encoding=ENCODING) as json_file:
        try:  # Python 3
            json_file.write(json.dumps(RESULTS_HISTORY))
        except TypeError:  # Python 2
            json_file.write(json.dumps(RESULTS_HISTORY).decode(ENCODING))


def get_global_best(values):
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


def save_html_evaluations(show_plot=True):
    """Save results history (evaluations) to disk (interactive html file)."""
    load_results_history()

    if not RESULTS_HISTORY:
        Print.info("results history is empty")
        Print.warn(
            "Suggestion: evaluate your model using the commands "
            "'test', 'k-fold' or 'grid_search'"
        )
        return

    html_src = RESULT_HTML_SRC_FOLDER
    result_html_file = RESULT_HTML_OUT_FILE % CLF.__name__
    fout = open(result_html_file, 'w', encoding=ENCODING)
    fhtml = open(
        module_path(html_src + "model_evaluation.html"),
        'r', encoding=ENCODING
    )

    for line in fhtml.readlines():
        if "plotly.min.js" in line:
            plotly_path = module_path(html_src + "plotly.min.js")
            with open(plotly_path, 'r', encoding=ENCODING) as fplotly:
                fout.write(u'    <script type="text/javascript">')
                fout.write(fplotly.read())
                fout.write(u'</script>\n')

        elif "angular.min.js" in line:
            angular_path = module_path(html_src + "angular.min.js")
            with open(angular_path, 'r', encoding=ENCODING) as fangular:
                fout.write(u'    <script type="text/javascript">')
                fout.write(fangular.read())
                fout.write(u'</script>\n')

        elif "data.js" in line:
            fout.write(u'    <script type="text/javascript">')
            fout.write(u'var $model_name = "%s"; ' % CLF.get_name())
            fout.write(
                u'var $results = JSON.parse("%s");'
                %
                json.dumps(RESULTS_HISTORY).replace('"', r'\"')
            )
            fout.write(u'</script>\n')
        else:
            fout.write(line)

    fhtml.close()
    fout.close()

    if show_plot:
        webbrowser.open(result_html_file)
        # webbrowser.get('firefox').open(result_html_file)

    Print.info("saved in file '%s'" % result_html_file)


def get_results_history(path, method, def_cat):
    """Given a path, a method and a default category return results history."""
    load_results_history()
    return RESULTS_HISTORY[path][method][def_cat]


def is_in_cache(path, method, def_cat, s, l, p, a):
    """Return whether this evaluation is already computed."""
    s, l = round_fix(s), round_fix(l)
    p, a = round_fix(p), round_fix(a)
    results = get_results_history(path, method, def_cat)
    return results["accuracy"]["value"][s][l][p][a] != {}


def train(
    x_train, y_train, n_grams,
    train_path='', folder_label=None,
    save=True, leave_pbar=True
):
    """Train a new model with the given training set."""
    global ARGS_CATS

    if len(y_train):
        CLF.fit(x_train, y_train, n_grams, leave_pbar=leave_pbar)

        if save:
            CLF.save_model()
            ARGS_CATS = CLF.get_categories()
    else:
        Print.warn(
            "Suggestion: train %s %s %s"
            %
            (
                train_path,
                STR_FOLDER if not folder_label else STR_FILE,
                "%d-%s" % (n_grams, STR_NGRAMS) if n_grams > 1 else ''
            )
        )


def get_test_data_cache(path, def_cat, method, s, l, p, a):
    """Return test results from cache."""
    s, l = round_fix(s), round_fix(l)
    p, a = round_fix(p), round_fix(a)
    if is_in_cache(path, method, def_cat, s, l, p, a):
        Print.info("retrieving from cache")

        y_true = []
        y_pred = []

        rh = get_results_history(path, method, def_cat)
        cm = np.array(rh["confusion_matrix"][s][l][p][a][0])
        categories = rh["categories"]

        for icat_true, row in enumerate(cm):
            y_true.extend([icat_true] * sum(row))
            for icat_pred in range(len(row)):
                y_pred.extend([icat_pred] * row[icat_pred])

        return y_true, y_pred, categories
    return None, None, None


def load_data(
    data_path, folder_label, def_cat=None,
    return_cat_index=True, cmd_name=STR_TEST
):
    """Load documents from disk, return the x_data, y_data and categories."""
    categories = CLF.get_categories()

    try:
        x_data, y_data = Dataset.load_from_files(data_path, folder_label)
    except OSError:
        Print.error(ERROR_NSD % data_path, raises=GetTestDataError)

    y_cats = set(y_data)
    if len(y_cats) == 0:
        Print.warn(
            "Suggestion: %s %s %s%s"
            %
            (
                cmd_name, data_path,
                STR_FOLDER if not folder_label else STR_FILE,
                " ..." if cmd_name != STR_TEST else ''
            ), raises=GetTestDataError
        )

    unkown_cats = [cat for cat in y_cats if cat not in categories]
    if len(unkown_cats):
        Print.error(
            "Unknown categories: model excepts %s but received %s."
            %
            (
                ", ".join(["'%s'" % c for c in categories]),
                ", ".join(["'%s'" % c for c in unkown_cats])
            ), raises=GetTestDataError
        )

    if return_cat_index:
        y_data = [CLF.get_category_index(y) for y in y_data]

    if def_cat == STR_UNKNOWN:
        categories += [STR_UNKNOWN_CATEGORY]

    return x_data, y_data, categories


def save_results(
    rh, categories, accuracy, report, conf_matrix, k_fold, i_fold, s, l, p, a
):
    """Save evaluation results to disk."""
    rf = round_fix
    s, l = rf(s), rf(l)
    p, a = rf(p), rf(a)

    rh["categories"] = categories

    # if there aren't previous best results, initialize them to -1
    if rh["accuracy"]["best"]["value"] == {}:
        rh["accuracy"]["best"]["value"] = -1
        for metric, avg in product(METRICS, AVGS):
            if avg in report:  # scikit-learn > 0.20 does not include 'micro avg' in report
                rh[metric][avg]["best"]["value"] = -1
        for cat in categories:
            for metric in METRICS:
                rh[metric]["categories"][cat]["best"]["value"] = -1

    # if fold results array is empty, create new ones
    if rh["accuracy"]["fold_values"][s][l][p][a] == {}:
        rh["accuracy"]["fold_values"][s][l][p][a] = [0] * k_fold
        rh["confusion_matrix"][s][l][p][a] = [None] * k_fold
        for metric, avg in product(METRICS, AVGS):
            if avg in report:
                rh[metric][avg]["fold_values"][s][l][p][a] = [0] * k_fold
        for cat in categories:
            for metric in METRICS:
                rh[metric]["categories"][cat]["fold_values"][s][l][p][a] = [0] * k_fold

    # saving fold results
    rh["accuracy"]["fold_values"][s][l][p][a][i_fold] = rf(accuracy)
    for metric, avg in product(METRICS, AVGS):
        if avg in report:
            rh[metric][avg]["fold_values"][s][l][p][a][i_fold] = rf(report[avg][metric])
    for cat in categories:
        for metric in METRICS:
            rh[metric]["categories"][cat]["fold_values"][s][l][p][a][i_fold] = rf(
                report[cat][metric]
            )
    rh["confusion_matrix"][s][l][p][a][i_fold] = conf_matrix.tolist()

    # if this is the last fold, compute and store averages and best values
    if i_fold + 1 == k_fold:
        accuracy_avg = rf(mean(rh["accuracy"]["fold_values"][s][l][p][a]))
        rh["accuracy"]["value"][s][l][p][a] = accuracy_avg

        best_acc = rh["accuracy"]["best"]
        if accuracy_avg > best_acc["value"]:
            best_acc["value"] = accuracy_avg
            best_acc["s"], best_acc["l"] = s, l
            best_acc["p"], best_acc["a"] = p, a

        for metric, avg in product(METRICS, AVGS):
            if avg in report:
                metric_avg = rf(mean(rh[metric][avg]["fold_values"][s][l][p][a]))
                rh[metric][avg]["value"][s][l][p][a] = metric_avg

                best_metric_avg = rh[metric][avg]["best"]
                if metric_avg > best_metric_avg["value"]:
                    best_metric_avg["value"] = metric_avg
                    best_metric_avg["s"], best_metric_avg["l"] = s, l
                    best_metric_avg["p"], best_metric_avg["a"] = p, a

        for cat in categories:
            for metric in METRICS:
                metric_cat_avg = rf(mean(
                    rh[metric]["categories"][cat]["fold_values"][s][l][p][a]
                ))
                rh[metric]["categories"][cat]["value"][s][l][p][a] = metric_cat_avg

                best_metric_cat = rh[metric]["categories"][cat]["best"]
                if metric_cat_avg > best_metric_cat["value"]:
                    best_metric_cat["value"] = metric_cat_avg
                    best_metric_cat["s"], best_metric_cat["l"] = s, l
                    best_metric_cat["p"], best_metric_cat["a"] = p, a

    save_results_history()


def results(
    y_true, y_pred, categories, def_cat, cache=True, method="test",
    data_path='', folder=False, plots=True, k_fold=1, i_fold=0
):
    """Compute evaluation results and save them to disk."""
    import warnings
    warnings.filterwarnings('ignore')
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

    unknown_cati = len(categories) - 1
    unclassified = None
    if data_path and def_cat == STR_UNKNOWN:
        unclassified = sum(map(lambda v: v == unknown_cati, y_pred))

    if data_path and unclassified:
        cat_acc = []
        for cat in CLF.get_categories():
            cat_acc.append((
                cat,
                accuracy_score(
                    [
                        CLF.get_category_index(cat) if y == unknown_cati else y
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
        Print.warn(
            "Suggestion: test %s %s %s"
            %
            (
                data_path,
                STR_FOLDER if folder else STR_FILE,
                best_acc[0]
            )
        )

    Print.show()

    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=range(len(categories)), target_names=categories,
        output_dict=True
    )

    s, l, p, a = CLF.get_hyperparameters()

    if not cache or not is_in_cache(data_path, method, def_cat, s, l, p, a):
        save_results(
            get_results_history(data_path, method, def_cat),
            categories, accuracy, report,
            conf_matrix, k_fold, i_fold,
            s, l, p, a
        )

    if plots:
        plot_confusion_matrices(
            [conf_matrix], categories,
            r"$\sigma=%.3f; \lambda=%.3f; \rho=%.3f; \alpha=%.3f$"
            %
            (s, l, p, a)
        )

    warnings.filterwarnings('default')


def evaluations_info(data_path=None, method=None):
    """Print evaluations best values."""
    load_results_history()
    rh = RESULTS_HISTORY
    ps = Print.style

    for dpath in rh:
        if data_path and dpath != data_path:
            continue

        for md in rh[dpath]:

            if method and md != method:
                continue

            for def_cat in rh[dpath][md]:
                Print.show("\n%s %s %s %s " % (
                    ps.fail(">"), ps.green(md),
                    ps.blue(dpath), ps.blue(def_cat)
                ), False)

                evl = rh[dpath][md][def_cat]["accuracy"]["value"]
                n_evl = len([
                    a for s in evl for l in evl[s]
                    for p in evl[s][l] for a in evl[s][l][p]
                ])
                Print.show("(%d evaluations)" % n_evl)

                best = rh[dpath][md][def_cat]["accuracy"]["best"]
                Print.show(
                    "Best %s: %s %s" % (
                        ps.green("accuracy"), ps.warning(best["value"]),
                        ps.blue("(s %s l %s p %s a %s)") % (
                            best["s"], best["l"], best["p"], best["a"]
                        )
                    ),
                    offset=4
                )
                for metric in sorted(rh[dpath][md][def_cat].keys()):

                    if metric not in EXCP_METRICS:
                        Print.show("Best %s:" % ps.green(metric), offset=4)
                        rh_metric = rh[dpath][md][def_cat][metric]
                        for cat in rh_metric["categories"]:
                            best = rh_metric["categories"][cat]["best"]
                            Print.show("%s: %s %s" % (
                                cat, ps.warning(best["value"]),
                                ps.blue("(s %s l %s p %s a %s)") % (
                                    best["s"], best["l"], best["p"], best["a"]
                                )
                            ), offset=8)

                        Print.show(ps.header(ps.bold("Averages:")), offset=8)
                        for avg in rh_metric:
                            if avg != "categories":
                                best = rh_metric[avg]["best"]
                                Print.show("%s: %s %s" % (
                                    ps.header(avg), ps.warning(best["value"]),
                                    ps.blue("(s %s l %s p %s a %s)")
                                    % (
                                        best["s"], best["l"],
                                        best["p"], best["a"]
                                    )
                                ), offset=10)

    print()


def delete_results_slpa(rh_metric, hparams, only_count=False, best=True):
    """Remove evaluations from history given hyperparameters s, l, p, a."""
    count = 0
    update_best = False
    hps = hparams
    if best:
        values = rh_metric["value"]
        best = rh_metric["best"]

        if best["s"] == hps["s"] or best["l"] == hps["l"] or \
           best["p"] == hps["p"] or best["a"] == hps["a"]:
            update_best = True
    else:
        values = rh_metric
    ss = list(values.keys())
    for s in ss:
        if hps["s"] is not None and s != hps["s"]:
            continue
        ll = list(values[s].keys())
        for l in ll:
            if hps["l"] is not None and l != hps["l"]:
                continue
            pp = list(values[s][l].keys())
            for p in pp:
                if hps["p"] is not None and p != hps["p"]:
                    continue
                aa = list(values[s][l][p].keys())
                for a in aa:
                    if hps["a"] is not None and a != hps["a"]:
                        continue

                    if not only_count:
                        del values[s][l][p][a]

                    count += 1

                if not values[s][l][p]:
                    del values[s][l][p]

            if not values[s][l]:
                del values[s][l]

        if not values[s]:
            del values[s]

    if update_best and not only_count:
        rh_metric["best"] = get_global_best(values)

    return count


def delete_results(data_path, method, def_cat, hparams, only_count=False):
    """Remove evaluations from history."""
    Print.verbosity_region_begin(VERBOSITY.QUIET)
    load_results_history()
    Print.verbosity_region_end()

    rh = RESULTS_HISTORY
    hps = hparams
    count_details = RecursiveDefaultDict()
    count = 0
    dpaths = list(rh.keys())
    for dpath in dpaths:
        if data_path and dpath != data_path:
            continue
        methods = list(rh[dpath].keys())
        for md in methods:
            if method and md != method:
                continue
            def_cats = list(rh[dpath][md].keys())
            for dc in def_cats:
                if def_cat and dc != def_cat:
                    continue
                rh_accuracy = rh[dpath][md][dc]["accuracy"]
                count_details[dpath][md][dc] = delete_results_slpa(
                    rh_accuracy, hps, only_count
                )
                count += count_details[dpath][md][dc]

                if not only_count:
                    delete_results_slpa(
                        rh[dpath][md][dc]["confusion_matrix"], hps, best=False
                    )

                if not rh[dpath][md][dc]["confusion_matrix"]:
                    del rh[dpath][md][dc]

                metrics = list(rh[dpath][md][dc].keys())
                for metric in metrics:
                    if metric not in EXCP_METRICS:
                        rh_metric = rh[dpath][md][dc][metric]
                        categories = list(rh_metric["categories"].keys())
                        for cat in categories:
                            if not only_count:
                                delete_results_slpa(
                                    rh_metric["categories"][cat],
                                    hps
                                )
                        avgs = list(rh_metric.keys())
                        for avg in avgs:
                            if avg != "categories":
                                rh_avg = rh[dpath][md][dc][metric][avg]
                                if not only_count:
                                    delete_results_slpa(rh_avg, hps)

                if not rh[dpath][md][dc]:
                    del rh[dpath][md][dc]

            if not rh[dpath][md]:
                del rh[dpath][md]

        if not rh[dpath]:
            del rh[dpath]

    return count, count_details


def evaluations_remove(data_path, method, def_cat, hparams):
    """Evaluation remove command handler."""
    global RESULTS_HISTORY
    count, count_details = delete_results(
        data_path, method, def_cat, hparams, only_count=True
    )

    if count > 0:
        print()
        Print.warn(
            "A total of %d evaluation(s) will be %s" %
            (count, Print.style.bold("removed")), False
        )
        Print.warn(". Details below:", decorator=False)
        print()
    else:
        Print.info("nothing to be removed")
        return

    for dp in count_details:
        for md in count_details[dp]:
            for dc in count_details[dp][md]:
                if count_details[dp][md][dc] > 0:
                    Print.show(Print.style.blue(
                        "  %s %s %s: " % (dp, md, dc)),
                        False
                    )
                    Print.show(Print.style.bold(Print.style.fail(
                        "%d evaluation(s)" % (count_details[dp][md][dc])
                    )))

    print()
    Print.warn("Do you %s" % Print.style.bold("really"), False)
    Print.warn(" want to proceed? [Y/n] ", False, decorator=False)
    try:
        if input() == 'Y':
            delete_results(data_path, method, def_cat, hparams)
            if RESULTS_HISTORY:
                Print.verbosity_region_begin(VERBOSITY.QUIET)
                save_results_history()
                Print.verbosity_region_end()
            else:
                rh_file = path.join(
                    CLF.__models_folder__,
                    CLF.get_name() + RESULT_HISTORY_EXT
                )
                if path.exists(rh_file):
                    remove_file(rh_file)
            Print.info("%d items were removed" % count)
    except EOFError:
        pass
    print()


def test(test_path, folder_label, def_cat, s, l, p, a, cache):
    """Test the model with a given test set."""
    try:
        y_pred = None

        CLF.set_hyperparameters(s, l, p, a)

        if cache:
            y_test, y_pred, categories = get_test_data_cache(
                test_path, def_cat, STR_TEST, s, l, p, a
            )

        if not y_pred:
            x_test, y_test, categories = load_data(
                test_path, folder_label, def_cat
            )

            try:
                y_pred = CLF.predict(x_test, def_cat, labels=False)
            except InvalidCategoryError:
                Print.error(ERROR_ICN % def_cat)
                return

        results(
            y_test, y_pred, categories,
            def_cat, cache, STR_TEST,
            test_path, folder_label
        )
    except GetTestDataError:
        pass


def k_fold_validation(
    data_path, folder_label, def_cat, n_grams, k_fold, s, l, p, a, cache=True
):
    """Perform a stratified k-fold cross validation using the given data."""
    global CLF

    try:
        x_data, y_data, categories = load_data(
            data_path, folder_label, def_cat,
            return_cat_index=False, cmd_name="k_fold"
        )
    except GetTestDataError:
        return

    Print.set_verbosity(VERBOSITY.NORMAL)
    model_name = CLF.get_name()
    method = k_fold2method(k_fold)

    x_data, y_data = np.array(x_data), np.array(y_data)
    skf = StratifiedKFold(n_splits=k_fold)
    progress_bar = tqdm(total=k_fold, desc=" K-Fold Progress")
    try:
        for i_fold, (train_ix, test_ix) in enumerate(skf.split(x_data, y_data)):
            if not cache or not is_in_cache(
                data_path, method, def_cat, s, l, p, a
            ):
                x_train, y_train = x_data[train_ix], y_data[train_ix]
                y_test = [CLF.get_category_index(y) for y in y_data[test_ix]]
                x_test = x_data[test_ix]

                CLF = SS3(name=model_name)
                CLF.set_hyperparameters(s, l, p, a)
                train(x_train, y_train, n_grams, save=False, leave_pbar=False)

                try:
                    y_pred = CLF.predict(
                        x_test, def_cat, labels=False, leave_pbar=False
                    )
                except InvalidCategoryError:
                    Print.error(ERROR_ICN % def_cat)
                    return

                results(
                    y_test, y_pred,
                    categories, def_cat,
                    cache, method, data_path,
                    plots=False, k_fold=k_fold, i_fold=i_fold
                )

            progress_bar.update(1)
    except KeyboardInterrupt:
        Print.set_verbosity(VERBOSITY.VERBOSE)
        print()
        Print.warn("Interrupted!")

    progress_bar.close()
    CLF = SS3(name=model_name)
    CLF.load_model()
    Print.set_verbosity(VERBOSITY.VERBOSE)

    Print.show()
    k_fold_classification_report(
        data_path, method, def_cat,
        s, l, p, a
    )


def grid_search_loop(
    data_path, x_test, y_test, categories, def_cat,
    k_fold, i_fold, ss, ll, pp, aa, cache=True, leave_pbar=True
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

    method = k_fold2method(k_fold)
    S, L, P, _ = CLF.get_hyperparameters()

    Print.verbosity_region_begin(VERBOSITY.QUIET)
    try:
        for s, l, p in slp_list:
            CLF.set_hyperparameters(s, l, p)
            updated = False
            for a in aa:
                if not cache or not is_in_cache(
                    data_path, method, def_cat, s, l, p, a
                ):
                    if not updated:
                        progress_desc.set_description_str(
                            " Status: [updating model...] "
                            "(s=%.3f; l=%.3f; p=%.3f; a=%.3f)"
                            %
                            (s, l, p, a)
                        )
                        CLF.update_values()
                        updated = True

                    CLF.set_alpha(a)
                    progress_desc.set_description_str(
                        " Status: [classifying...] "
                        "(s=%.3f; l=%.3f; p=%.3f; a=%.3f)"
                        %
                        (s, l, p, a)
                    )

                    try:
                        y_pred = CLF.predict(
                            x_test, def_cat, labels=False, leave_pbar=False
                        )
                    except InvalidCategoryError:
                        Print.error(ERROR_ICN % def_cat)
                        return

                    results(
                        y_test, y_pred,
                        categories, def_cat,
                        cache, method, data_path,
                        plots=False, k_fold=k_fold, i_fold=i_fold
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
    except KeyboardInterrupt:
        Print.set_verbosity(VERBOSITY.VERBOSE)
        print()
        Print.warn("Interrupted!")

    progress_desc.set_description_str(" Status: [finished]")
    progress_bar.close()
    progress_desc.close()

    CLF.set_hyperparameters(S, L, P)
    CLF.update_values()

    Print.verbosity_region_end()


def grid_search(
    data_path, folder_label, def_cat, n_gram, k_fold,
    ss, ll, pp, aa, cache=True
):
    """Perform a grid search using values from `ss`, ``ll``, ``pp``, ``aa``."""
    global CLF

    try:
        print()
        if not k_fold:  # if test
            try:
                x_data, y_data, categories = load_data(
                    data_path, folder_label, def_cat,
                    cmd_name="grid_search"
                )
            except GetTestDataError:
                return
            x_test, y_test = x_data, y_data
            grid_search_loop(
                data_path, x_test, y_test, categories, def_cat,
                1, 0, ss, ll, pp, aa, cache
            )
        else:  # if k-fold

            try:
                x_data, y_data, categories = load_data(
                    data_path, folder_label, def_cat,
                    return_cat_index=False, cmd_name="grid_search"
                )
            except GetTestDataError:
                return

            Print.set_verbosity(VERBOSITY.NORMAL)
            model_name = CLF.get_name()
            s, l, p, a = CLF.get_hyperparameters()

            x_data, y_data = np.array(x_data), np.array(y_data)
            skf = StratifiedKFold(n_splits=k_fold)
            progress_bar = tqdm(
                position=0, total=k_fold,
                desc=" K-Fold Progress"
            )
            try:
                for i_fold, (train_ix, test_ix) in enumerate(
                    skf.split(x_data, y_data)
                ):
                    x_train, y_train = x_data[train_ix], y_data[train_ix]
                    y_test = [CLF.get_category_index(y) for y in y_data[test_ix]]
                    x_test = x_data[test_ix]

                    CLF = SS3(name=model_name)
                    train(x_train, y_train, n_gram, save=False, leave_pbar=False)

                    grid_search_loop(
                        data_path, x_test, y_test, categories, def_cat,
                        k_fold, i_fold, ss, ll, pp, aa, cache, leave_pbar=False
                    )

                    save_results_history()

                    progress_bar.update(1)
            except KeyboardInterrupt:
                Print.set_verbosity(VERBOSITY.VERBOSE)
                print()
                Print.warn("Interrupted!")

            progress_bar.close()
            CLF = SS3(name=model_name)
            CLF.load_model()
            CLF.set_hyperparameters(s, l, p, a)
            Print.set_verbosity(VERBOSITY.VERBOSE)

        Print.warn(
            "Suggestion: use the command 'plot %s' to visualize the results"
            % STR_EVALUATIONS
        )
        print("\n")
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt")


class SS3Prompt(Cmd):
    """Prompt main class."""

    @requires_args
    def do_new(self, args):
        """
        Create a new empty SS3 model with a given name.

        usage:
            new MODEL_NAME

        required arguments:
         MODEL_NAME      the model's name
        """
        global CLF, RESULTS_HISTORY
        args = split_args(args)
        model_name = args[0].lower()

        if args:
            if model_name in MODELS:
                print()
                Print.warn(WARN_OVERWRITE, False)
                if input() == 'Y':
                    rh_file = path.join(
                        SS3.__models_folder__,
                        model_name + RESULT_HISTORY_EXT
                    )
                    if path.exists(rh_file):
                        remove_file(rh_file)

                    model_file = path.join(
                        SS3.__models_folder__,
                        "%s.%s" % (model_name, STR_MODEL_EXT)
                    )
                    if path.exists(model_file):
                        remove_file(model_file)
                    Print.info("model has been overwritten")
                    print()
                else:
                    print()
                    return

            RESULTS_HISTORY = None
            CLF = SS3(name=model_name)
        else:
            Print.error(
                "Empty model's name: please provide your model's name"
            )

    @requires_args
    def do_load(self, args):
        """
        Load a local model (given its name).

        usage:
            load MODEL_NAME

        required arguments:
         MODEL_NAME      the model's name
        """
        global CLF, ARGS_CATS, RESULTS_HISTORY
        args = split_args(args)

        try:
            new_clf = SS3(name=args[0])
            new_clf.load_model()

            CLF = new_clf
            ARGS_CATS = CLF.get_categories()
            RESULTS_HISTORY = None
        except IOError:
            Print.error(
                "Failed to load the model: "
                "No model named '%s' was found in folder ./%s"
                % (args[0], SS3.__models_folder__)
            )

    @requires_model
    def do_rename(self, args):
        """
        Rename the current model with a given name.

        usage:
            rename NEW_MODEL_NAME

        required arguments:
         NEW_MODEL_NAME      the model's new name
        """
        args = split_args(args)

        if len(args) == 1:
            m_folder = CLF.__models_folder__
            m_name = CLF.__name__
            rh_ext = RESULT_HISTORY_EXT
            rh_file = path.join(m_folder, m_name + rh_ext)
            rh_new_file = path.join(m_folder, args[0] + rh_ext)

            rename = True
            if path.exists(rh_new_file):
                print()
                Print.warn(WARN_OVERWRITE, False)
                if input() != 'Y':
                    rename = False

            if rename:
                rename_file(rh_file, rh_new_file)

                m_ext = STR_MODEL_EXT
                model_file = path.join(m_folder, "%s.%s" % (m_name, m_ext))
                model_new_file = path.join(
                    m_folder, "%s.%s" % (args[0], m_ext)
                )
                rename_file(model_file, model_new_file)
                CLF.__name__ = args[0]
        else:
            Print.error(ERROR_WAN % (1, len(args)))

    @requires_model
    def do_clone(self, args):
        """
        Create a copy of the current model with a given name.

        usage:
            clone NEW_MODEL_NAME

        required arguments:
         NEW_MODEL_NAME      the new model's name
        """
        args = split_args(args)

        if len(args) == 1:
            load_results_history()
            CLF.__name__ = args[0]
            CLF.save_model()
            save_results_history()
        else:
            Print.error(ERROR_WAN % (1, len(args)))

    @requires_model
    @requires_args
    def do_train(self, args):
        """
        Train the model using a training set and then save it.

        usage:
            train TRAIN_PATH [LABEL] [N-gram]

        required arguments:
         TRAIN_PATH     the training set path

        optional arguments:
         LABEL          where to read category labels from.
                        values:{file,folder} (default: folder)

         N-grams        indicates the maximum n-grams to be learned (e.g. a
                        value of "1-grams" means only words will be learned;
                        "2-grams" only 1-grams and 2-grams;
                        "3-grams", only 1-grams, 2-grams and 3-grams;
                        and so on).
                        value: {N-grams} with N integer > 0 (default: 1-grams)

        examples:
         train a/training/set/path 3-grams
        """
        try:
            train_path, folder_label, n_grams = self.args_train(args)
            try:
                x_train, y_train = Dataset.load_from_files(train_path, folder_label, True)
            except OSError:
                Print.error(ERROR_NSD % train_path)
                return

            train(x_train, y_train, n_grams, train_path, folder_label)

        except ArgsParseError:
            pass

    @requires_model
    @requires_args
    def do_k_fold(self, args):
        """
        Perform a stratified k-fold validation using the given dataset set.

        usage:
            k_fold PATH [LABEL] [DEF_CAT] [N-grams] [N-fold] [P VAL ...] [no-cache]

        required arguments:
         PATHthe    dataset path

        optional arguments:
         LABEL      where to read category labels from.
                    values:{file,folder} (default: folder)

         DEF_CAT    default category to be assigned when the model is not
                    able to actually classify a document.
                    values: {most-probable,unknown} or a category label
                    (default: most-probable)

         N-grams    indicates the maximum n-grams to be learned (e.g. a
                    value of "1-grams" means only words will be learned;
                    "2-grams" only 1-grams and 2-grams;
                    "3-grams", only 1-grams, 2-grams and 3-grams;
                    and so on).
                    value: {N-grams} with N integer > 0 (default: 1-grams)

         K-fold     indicates the number of folds to be used.
                    value: {K-fold} with K integer > 1 (default: 4-fold)

         P VAL      sets a hyperparameter value (e.g. s 0.45)
                    P values: {s,l,p,a}
                    VAL values: float

         no-cache   if present, disable the cache and recompute values

        examples:
         k_fold a/dataset/path 10-fold
         k_fold a/dataset/path 4-fold -s .45 -l 1.1 -p 1
        """
        try:
            data_path, folder_label, def_cat,\
                n_grams, k_fold, hparams, cache = self.args_k_fold(args)

            s, l, p, a = CLF.get_hyperparameters()
            k_fold_validation(
                data_path, folder_label, def_cat, n_grams, k_fold,
                hparams["s"], hparams["l"], hparams["p"], hparams["a"],
                cache
            )
            CLF.set_hyperparameters(s, l, p, a)
        except ArgsParseError:
            return

    @requires_model
    @requires_args
    def do_test(self, args):
        """
        Test the model using the given test set.

        usage:
            test TEST_PATH [LABEL] [DEF_CAT] [P VAL ...] [no-cache]

        required arguments:
         TEST_PATH  the test set path

        optional arguments:
         LABEL      where to read category labels from.
                    values:{file,folder} (default: folder)

         DEF_CAT    default category to be assigned when the model is not
                    able to actually classify a document.
                    values: {most-probable,unknown} or a category label
                    (default: most-probable)

         P VAL      sets a hyperparameter value
                    examples: s .45; s .5;
                    P values: {s,l,p,a}
                    VAL values: float

         no-cache   if present, disable the cache and recompute values

        examples:
         test a/testset/path
         test a/testset/path -s .45 -l 1.1 -p 1
         test a/testset/path unknown -s .45 -l 1.1 -p 1 no-cache
        """
        try:
            test_path, folder_label,\
                def_cat, hparams, cache = self.args_test(args)

            s, l, p, a = CLF.get_hyperparameters()
            test(
                test_path, folder_label, def_cat,
                hparams["s"], hparams["l"], hparams["p"], hparams["a"],
                cache
            )
            CLF.set_hyperparameters(s, l, p, a)
        except ArgsParseError:
            return

    @requires_model
    def do_live_test(self, args):
        """
        Interactively and graphically test the model.

        usage:
            live_test [TEST_PATH [LABEL]] [verbose]

        optional arguments:
         TEST_PATH  the test set path

         LABEL      where to read category labels from.
                    values: {file,folder} (default: folder)

         verbose    if present, run in verbose mode

        examples:
         live_test
         live_test a/testset/path
         live_test a/testset/path verbose
        """
        try:
            test_path, folder_label, verbose = self.args_live_test(args)
        except ArgsParseError:
            return

        if test_path:
            try:
                success = Server.set_testset_from_files(test_path, folder_label)
                if not success:
                    Print.warn(
                        "Suggestion: live_test %s %s" % (
                            test_path,
                            STR_FOLDER if not folder_label else STR_FILE
                        )
                    )
                    return
            except OSError:
                Print.error(ERROR_NSD % test_path)
                return
        else:
            Server.set_testset([], [])

        Server.serve(CLF, quiet=not verbose)

    @requires_model
    @requires_args
    def do_grid_search(self, args):
        """
        Given a dataset, perform a grid search using the given hyperparameters values.

        usage:
            grid_search PATH [LABEL] [DEF_CAT] [METHOD] P EXP [P EXP ...] [no-cache]

        required arguments:
         PATH       the dataset path
         P EXP      a list of values for a given hyperparameter.
                    where:
                     P    is a hyperparameter name. values: {s,l,p,a}
                     EXP  is a python expression returning a float or
                          a list of floats. Note: if this expression
                          contains whitespaces, use quotations marks
                          (e.g. "[0.5, 1.5]")
                    examples:
                     s [.3,.4,.5]
                     s "[.3, .4, .5]" (Note the whitespaces and the "")
                     p r(.2,.8,6)     (i.e. 6 points between .2 to .8)

        optional arguments:
         LABEL      where to read category labels from.
                    values:{file,folder} (default: folder)

         DEF_CAT    default category to be assigned when the model is not
                    able to actually classify a document.
                    values: {most-probable,unknown} or a category label
                    (default: most-probable)

         METHOD     the method to be used
                    values: {test, K-fold} (default: test)
                    where:
                      K-fold  indicates the number of folds to be used.
                              K is an integer > 1 (e.g 4-fold, 10-fold, etc.)

         no-cache   if present, disable the cache and recompute all the values

        examples:
         grid_search a/testset/path s r(.2,.8,6) l r(.1,2,6) -p r(.5,2,6) a [0,.01]
         grid_search a/dataset/path 4-fold -s [.2,.3,.4,.5] -l [.5,1,1.5] -p r(.5,2,6)
        """
        try:
            data_path, folder_label, def_cat,\
                n_grams, k_fold, hparams, cache = self.args_grid_search(args)
            grid_search(
                data_path, folder_label, def_cat, n_grams, k_fold,
                hparams["s"], hparams["l"], hparams["p"], hparams["a"],
                cache
            )
        except ArgsParseError:
            pass

    @requires_model
    def do_evaluations(self, args):
        """
        Perform different actions linked to evaluations results.

        usage:
            evaluations OPTION [PATH] [METHOD] [DEF_CAT] [P VAL [P VAL ...]

        required arguments:
         OPTION     indicates the action to perform
                    values: {info,plot,save,remove} (default: info)
                        info - show information about evaluations (including
                               best values).
                        plot - show an interactive 3-D plot with evaluation
                               results in the web browser (it also save it
                               to disk).
                        save - save the interactive 3-D plot to disk.
                        remove - delete evaluations results from history

        optional arguments:
         PATH       the dataset path used in the evaluate of interest

         METHOD     the method that was used in the evaluate of interest
                    values: {test,K-fold} where K is an integer > 1

         DEF_CAT    default category used in the evaluate of interest
                    values: {most-probable,unknown} or a category label

         P VAL      the hyperparameter value (only for option "remove")
                    P values: {s,l,p,a}
                    VAL values: float

        examples:
         * show information about all evaluations:
            evaluations info

         * show information about evaluations in path "a/dataset/path":
            evaluations info a/dataset/path

         * information about 3-fold evaluations in path "a/dataset/path":
            evaluations info a/dataset/path 3-fold

         * information about test evaluations in path "a/dataset/path":
            evaluations info a/dataset/path test

         * plot evaluations:
            evaluations plot

         * save evaluations:
            evaluations save

         * remove all evaluation result(s) in path "a/dataset/path":
            evaluations remove a/dataset/path

         * remove 4-fold evaluation result(s) in path "a/dataset/path"
           with l = 1.1 and s = .45:
            evaluations remove a/dataset/path 4-fold l 1.1 s .45
        """
        try:
            cmd, data_path, method,\
                def_cat, hparams = self.args_evaluations(args)
        except ArgsParseError:
            return

        if cmd == STR_INFO:
            evaluations_info(data_path, method)
        elif cmd == STR_PLOT:
            save_html_evaluations()
        elif cmd == STR_SAVE:
            save_html_evaluations(False)
        elif cmd == STR_REMOVE:
            evaluations_remove(data_path, method, def_cat, hparams)
        else:
            Print.error(ERROR_UA % cmd)

    @requires_model
    def do_classify(self, args):
        """
        Classify a document.

        usage:
            classify [DOCUMENT_PATH]

        optional arguments:
         DOCUMENT_PATH   the path to the document file
        """
        try:
            document = self.args_classify(args)
        except ArgsParseError:
            return

        result = CLF.classify(document)

        print()
        print("SS3 prediction is:\n")
        for i, catinfo in enumerate(result):
            if catinfo[1]:
                cat_result = " %d. %s (confidence value is %.1f)" % (
                    i + 1,
                    CLF.get_category_name(catinfo[0]).upper(),
                    catinfo[1]
                )
                if i == 0:
                    cat_result = Print.style.bold(cat_result)
                print(cat_result)
        print()

    @requires_model
    @requires_args
    def do_learn(self, args):
        """
        Learn a new document.

        usage:
            learn CAT [N-grams] [DOCUMENT_PATH]

        required arguments:
         CAT            the category label

        optional arguments:
         N-grams        indicates the maximum n-grams to be learned (e.g. a
                        value of "1-grams" means only words will be learned;
                        "2-grams" only 1-grams and 2-grams;
                        "3-grams", only 1-grams, 2-grams and 3-grams;
                        and so on).
                        value: {N-grams} with N integer > 0 (default: 1-grams)

         DOCUMENT_PATH   the path to the document file
        """
        global ARGS_CATS

        try:
            cat, n_grams, document = self.args_learn(args)

            if document.strip():
                CLF.learn(document, cat, max_depth=n_grams)
                ARGS_CATS = CLF.get_categories()
            else:
                Print.info("empty document")
        except ArgsParseError:
            pass

    @requires_model
    def do_update(self, args):
        """Update model values (cv, gv, lv, etc.)."""
        CLF.update_values()
        Print.warn(
            "Remember to use the 'save' command if you want these changes "
            "to be permanently stored"
        )

    @requires_model
    def do_save(self, args):
        """
        Save to disk the model, learned vocabulary, evaluations results, etc.

        usage:
            save OPTION

        required arguments:
         OPTION     indicates what to save to disk
                    values:
                        model; (default)
                        evaluations;
                        vocabulary [CAT];
                        stopwords [SG_THRESHOLD];

                        where:
                         CAT           the category label

                         SG_THRESHOLD  significance (sg) value used as a
                                       threshold to consider words as
                                       stopwords (i.e. words with
                                       sg < ``sg_threshold`` for all
                                       categories will be considered as
                                       "stopwords")
                                       (default: .01)

        examples:
         save
         save model
         save vocabulary
         save vocabulary a_category
         save stopwords
         save stopwords .1
        """
        try:
            arg, value = self.args_save(args)
        except ArgsParseError:
            return

        # case 'model'
        if arg == STR_MODEL:
            CLF.save_model()

        # case 'vocabulary'
        elif arg == STR_VOCABULARY:
            if value:
                try:
                    CLF.save_cat_vocab(value)
                except InvalidCategoryError:
                    Print.error(ERROR_ICN % value)
            else:
                CLF.save_vocab()

        # case 'evaluations'
        elif arg == STR_EVALUATIONS:
            save_html_evaluations(False)

        # case 'stopwords'
        elif arg == STR_STOPWORDS:
            if value:
                stopwords = CLF.get_stopwords(value)
            else:
                stopwords = CLF.get_stopwords()

            if stopwords:
                stopwords_file = STOPWORDS_FILE % CLF.__name__
                with open(stopwords_file, "w", encoding=ENCODING) as fstopws:
                    fstopws.write(u'\n'.join(stopwords))
                Print.info("stopwords saved in '%s'" % stopwords_file)
            else:
                Print.warn(WARN_NO_STOPWORDS)

    @requires_model
    def do_info(self, args):
        """
        Show useful information.

        usage:
            info OPTION

        required arguments:
         OPTION     indicates what information to show
                    values: {all, parameters, categories, evaluations}
                            (default: all)

        examples:
         info
         info evaluations
        """
        args = split_args(args)

        if args and args[0] == STR_EVALUATIONS:
            evaluations_info()
        else:
            CLF.print_model_info()
            all_on = not args or args[0] == STR_ALL
            if all_on or args[0] == STR_PARAMETERS:
                CLF.print_hyperparameters_info()
            if all_on or args[0] == STR_CATEGORIES:
                CLF.print_categories_info()

    @requires_model
    @requires_args
    def do_debug_term(self, args):
        """
        Show debugging information about a given n-gram.

        Namely, print the n-gram frequency (fr), local value (lv),
        global value (gv), confidence value (cv), sanction (sn) weight and
        significance (sg) weight.

        usage:
            debug_term N_GRAM

        required arguments:
         N_GRAM     the n-gram (word, bigram, trigram, etc.) to debug

        examples:
         debug_term the
         debug_term potato
         debug_term "machine learning"
         debug_term "self driving car"
        """
        args = args.strip('"\'')
        if args:
            CLF.print_ngram_info(args)
        else:
            Print.error(ERROR_WAN % (1, 0))

    @requires_model
    @requires_args
    def do_plot(self, args):
        """
        Plot word value distribution curve or the evaluation results.

        usage:
            plot OPTION

        required arguments:
         OPTION     indicates what to plot
                    values:
                        evaluations;
                        distribution CAT;

                        where:
                         CAT           the category label

        examples:
         plot distribution a_category
         plot evaluations
        """
        args = split_args(args)

        if args[0] == STR_DISTRIBUTION:
            try:
                CLF.plot_value_distribution(args[1])
            except InvalidCategoryError:
                Print.error(ERROR_ICN % args[1])
                return
            except IndexError:
                Print.error(ERROR_WAN % (2, 1))
                Print.warn(
                    "Suggestion: "
                    "add the category name (e.g. plot distribution %s)"
                    %
                    CLF.get_category_name(0)
                )
                return
        elif args[0] == STR_EVALUATIONS:  # if evaluations
            save_html_evaluations()
        else:
            Print.error(ERROR_UA % args[0])

    @requires_model
    @requires_args
    def do_set(self, args):
        """
        Set a given hyperparameter value.

        usage:
            set P VAL [P VAL ...]

        required arguments:
         P VAL      sets a hyperparameter value
                    examples: s .45; s .5;
                    P values: {s,l,p,a}
                    VAL values: float

        examples:
         set s .5
         set l 0.5
         set p 2
         set s .5 l 0.5 p 2
        """
        try:
            hparams = self.args_set(args)
        except ArgsParseError:
            return

        CLF.set_hyperparameters(
            hparams["s"], hparams["l"], hparams["p"], hparams["a"]
        )

        Print.warn("Remember to use the 'update' command to update the model")

    @requires_model
    @requires_args
    def do_get(self, args):
        """
        Get a given hyperparameter value.

        usage:
            get PARAM

        required arguments:
         PARAM      the hyperparameter name
                    values: {s,l,p,a}

        examples:
         get s
         get l
         get p
         get a
        """
        args = split_args(args)

        hp = args[0]

        if len(args) != 1:
            Print.error(ERROR_WAN % (1, len(args)))
            return

        # case s
        if hp == STR_S:
            value = CLF.get_smoothness()
        # case l
        elif hp == STR_L:
            value = CLF.get_significance()
        # case p
        elif hp == STR_P:
            value = CLF.get_sanction()
        # case a
        elif hp == STR_A:
            value = CLF.get_alpha()
        # otherwise
        else:
            Print.error(ERROR_UA % hp)
            return

        print(Print.style.warning(value))

    @requires_model
    @requires_args
    def do_next_word(self, args):
        """
        Show up to 3 possible words to follow after the given sentence.

        usage:
            next_word SENT

        required arguments:
         SENT     a sentence

        examples:
         next_word "the self driving"
         next_word "a machine learning"
        """
        if not args:
            Print.error(ERROR_WAN % (1, 0))
            return

        args = args.strip('"\'')
        print()
        for cat in CLF.get_categories():
            next_w = [
                "%s%s"
                %
                (
                    Print.style.green(w.upper()),
                    Print.style.blue("(%.1f%%)" % (P * 100))
                )
                for w, fr, P in CLF.get_next_words(args, cat, 3)
            ]
            if next_w:
                print(" %s: %s" % (Print.style.bold(cat), " ".join(next_w)))
        print()

    def do_license(self, args):
        """Print the license."""
        print(
            'The MIT License (MIT)\n\n'

            'Copyright (c) 2019 Sergio Burdisso (sergio.burdisso@gmail.com)\n\n'

            'Permission is hereby granted, free of charge, to any person obtaining a copy\n'
            'of this software and associated documentation files (the "Software"), to deal\n'
            'in the Software without restriction, including without limitation the rights\n'
            'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
            'copies of the Software, and to permit persons to whom the Software is\n'
            'furnished to do so, subject to the following conditions:\n\n'

            'The above copyright notice and this permission notice shall be included in all\n'
            'copies or substantial portions of the Software.\n\n'

            'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
            'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
            'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
            'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
            'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
            'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
            'SOFTWARE.\n\n'
        )

    def do_exit(self, args=''):
        """Quit the program."""
        print("Bye")
        if readline:
            readline.set_history_length(HISTFILE_SIZE)
            try:
                readline.write_history_file(HISTFILE)
            except IOError:
                pass
        raise SystemExit

    def complete_info(self, text, line, begidx, endidx):
        """Complete arguments for 'info' command."""
        return [a for a in ARGS["info"] if a.startswith(text)]

    def complete_save(self, text, line, begidx, endidx):
        """Complete arguments for 'save' command."""
        return [a for a in ARGS["save"] + ARGS_CATS if a.startswith(text)]

    def complete_load(self, text, line, begidx, endidx):
        """Complete arguments for 'load' command."""
        return [a for a in ARGS["load"] if a.startswith(text)]

    def complete_train(self, text, line, begidx, endidx):
        """Complete arguments for 'train' command."""
        return [a for a in ARGS["train"] if a.startswith(text)]

    def complete_test(self, text, line, begidx, endidx):
        """Complete arguments for 'test' command."""
        return [a for a in ARGS["test"] + ARGS["train"] + ARGS_CATS if a.startswith(text)]

    def complete_live_test(self, text, line, begidx, endidx):
        """Complete arguments for 'test' command."""
        return [a for a in ARGS["live_test"] if a.startswith(text)]

    def complete_learn(self, text, line, begidx, endidx):
        """Complete arguments for 'learn' command."""
        return [a for a in ARGS["learn"] + ARGS_CATS if a.startswith(text)]

    def complete_set(self, text, line, begidx, endidx):
        """Complete arguments for 'set' command."""
        return [a for a in ARGS["set"] if a.startswith(text)]

    def complete_plot(self, text, line, begidx, endidx):
        """Complete arguments for 'plot' command."""
        return [a for a in ARGS["plot"] + ARGS_CATS if a.startswith(text)]

    def complete_grid_search(self, text, line, begidx, endidx):
        """Complete arguments for 'grid_search' command."""
        return [
            a for a in
            ARGS["grid_search"] + ARGS["test"] + ARGS["train"] + ARGS_CATS
            if a.startswith(text)
        ]

    def complete_evaluations(self, text, line, begidx, endidx):
        """Complete arguments for 'grid_search' command."""
        return [a for a in ARGS["evaluations"] if a.startswith(text)]

    def args_train(self, args):
        """Parse train arguments."""
        args = split_args(args)
        n_grams = 1
        folder_label = True

        if not args:
            Print.error(ERROR_WAN % (1, 0), raises=ArgsParseError)

        op_args = args[1:]
        op_args_ix = []

        folder_arg = intersect([STR_FILE, STR_FOLDER], op_args)
        if folder_arg:
            folder_label = folder_arg[0] == STR_FOLDER
            op_args_ix.append(op_args.index(folder_arg[0]))

        n_grams_arg = re_in(r"(.+)-" + STR_NGRAMS, op_args)
        if n_grams_arg:
            op_args_ix.append(op_args.index(n_grams_arg.group(0)))
            try:
                n_grams = int(n_grams_arg.group(1))
                if n_grams <= 0:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WNGRAM, raises=ArgsParseError)

        unkown_args = subtract(range(len(op_args)), op_args_ix)
        if len(unkown_args) > 0:
            Print.error(ERROR_WNAUA, raises=ArgsParseError)

        return args[0], folder_label, n_grams

    def args_test(self, args):
        """Parse test arguments."""
        args = split_args(args)
        def_cat = STR_MOST_PROBABLE
        folder_label = True
        cache = True

        if not CLF.get_categories():
            Print.error(ERROR_MNT, raises=ArgsParseError)

        if not args:
            Print.error(ERROR_WAN % (1, 0), raises=ArgsParseError)

        op_args = args[1:]
        op_args_ix = []

        folder_arg = intersect([STR_FILE, STR_FOLDER], op_args)
        if folder_arg:
            folder_label = folder_arg[0] == STR_FOLDER
            op_args_ix.append(op_args.index(folder_arg[0]))

        if STR_NO_CACHE in op_args:
            cache = False
            op_args_ix.append(op_args.index(STR_NO_CACHE))

        hparams, used_args_ix = parse_hparams_args(op_args)
        op_args_ix.extend(used_args_ix)

        def_cat_arg = subtract(range(len(op_args)), op_args_ix)
        if len(def_cat_arg) == 1:
            def_cat = op_args[def_cat_arg[0]]
        elif len(def_cat_arg) > 1:
            Print.error(
                ERROR_WAN % (len(op_args_ix) + 2, len(args)), raises=ArgsParseError
            )

        return args[0], folder_label, def_cat, hparams, cache

    def args_k_fold(self, args):
        """Parse k_fold arguments."""
        args = split_args(args)
        def_cat = STR_MOST_PROBABLE
        folder_label = True
        cache = True
        k_fold = 4
        n_grams = 1

        if not CLF.get_categories():
            Print.error(ERROR_MNT, raises=ArgsParseError)

        if len(args) < 1:
            Print.error(ERROR_WAN % (2, len(args)), raises=ArgsParseError)

        op_args = args[1:]
        op_args_ix = []

        if STR_NO_CACHE in op_args:
            cache = False
            op_args_ix.append(op_args.index(STR_NO_CACHE))

        n_grams_arg = re_in(r"(.+)-" + STR_NGRAMS, op_args)
        if n_grams_arg:
            op_args_ix.append(op_args.index(n_grams_arg.group(0)))
            try:
                n_grams = int(n_grams_arg.group(1))
                if n_grams <= 0:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WNGRAM, raises=ArgsParseError)

        k_fold_arg = re_in(r"(.+)-" + STR_FOLD, op_args)
        if k_fold_arg:
            op_args_ix.append(op_args.index(k_fold_arg.group(0)))
            try:
                k_fold = int(k_fold_arg.group(1))

                if k_fold < 2:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WKFOLD, raises=ArgsParseError)

        folder_arg = intersect([STR_FILE, STR_FOLDER], op_args)
        if folder_arg:
            folder_label = folder_arg[0] == STR_FOLDER
            op_args_ix.append(op_args.index(folder_arg[0]))

        hparams, used_args_ix = parse_hparams_args(op_args)
        op_args_ix.extend(used_args_ix)

        def_cat_arg = subtract(range(len(op_args)), op_args_ix)
        if len(def_cat_arg) == 1:
            def_cat = op_args[def_cat_arg[0]]
        elif len(def_cat_arg) > 1:
            Print.error(ERROR_WNAUA, raises=ArgsParseError)

        return args[0], folder_label, def_cat, n_grams, k_fold, hparams, cache

    def args_grid_search(self, args):
        """Parse grid_search arguments."""
        args = split_args(args)
        def_cat = STR_MOST_PROBABLE
        folder_label = True
        cache = True
        hparams = {}
        k_fold = 0
        n_grams = len(CLF.__max_fr__[0])

        if not CLF.get_categories():
            Print.error(ERROR_MNT, raises=ArgsParseError)

        if len(args) < 2:
            Print.error(ERROR_WAN % (2, len(args)), raises=ArgsParseError)

        op_args = args[1:]
        op_args_ix = []

        if STR_NO_CACHE in op_args:
            cache = False
            op_args_ix.append(op_args.index(STR_NO_CACHE))

        n_grams_arg = re_in(r"(.+)-" + STR_NGRAMS, op_args)
        if n_grams_arg:
            op_args_ix.append(op_args.index(n_grams_arg.group(0)))
            try:
                n_grams = int(n_grams_arg.group(1))
                if n_grams <= 0:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WNGRAM, raises=ArgsParseError)

        k_fold_arg = re_in(r"(.+)-" + STR_FOLD, op_args)
        if k_fold_arg:
            op_args_ix.append(op_args.index(k_fold_arg.group(0)))
            try:
                k_fold = int(k_fold_arg.group(1))

                if k_fold < 2:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WKFOLD, raises=ArgsParseError)

        folder_arg = intersect([STR_FILE, STR_FOLDER], op_args)
        if folder_arg:
            folder_label = folder_arg[0] == STR_FOLDER
            op_args_ix.append(op_args.index(folder_arg[0]))

        s, l, p, a = CLF.get_hyperparameters()
        no_hparams = True
        for key_args in ((STR_S, s), (STR_L, l), (STR_P, p), (STR_A, a)):
            hp_str, h_v = key_args
            arg = intersect(key_args, op_args)
            if arg:
                argi = op_args.index(arg[0])
                op_args_ix.extend([argi, argi + 1])
                try:
                    hparams[hp_str] = eval(op_args[argi + 1])
                except IndexError:
                    Print.error(ERROR_HVM % hp_str, raises=ArgsParseError)
                except BaseException:
                    Print.error(
                        "[python] error: "
                        "the value for the hyperparameter '%s' is not valid"
                        %
                        hp_str, raises=ArgsParseError
                    )

                # just in case the hyperparameter value is a single number
                try:
                    hparams[hp_str] = [float(hparams[hp_str])]
                except BaseException:
                    pass

                try:
                    hparams[hp_str] = [float(v) for v in hparams[hp_str]]
                except ValueError:
                    Print.error(
                        "Wrong hyperparameter value type: "
                        "Some of values for the hyperparameter '%s' are not numbers"
                        %
                        hp_str, raises=ArgsParseError
                    )

                no_hparams = False
            else:
                hparams[hp_str] = [h_v]

        if no_hparams:
            Print.error(
                "hyperparameters missing: at least one "
                "hyperparameter value range must be given",
                raises=ArgsParseError
            )

        def_cat_arg = subtract(range(len(op_args)), op_args_ix)
        if len(def_cat_arg) == 1:
            def_cat = op_args[def_cat_arg[0]]
            if def_cat not in [STR_MOST_PROBABLE, STR_UNKNOWN]:
                if CLF.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    Print.error(ERROR_ICN % def_cat, raises=ArgsParseError)
                    return
        elif len(def_cat_arg) > 1:
            Print.error(ERROR_WNAUA, raises=ArgsParseError)

        return args[0], folder_label, def_cat, n_grams, k_fold, hparams, cache

    def args_evaluations(self, args):
        """Parse evaluations arguments."""
        args = split_args(args)
        data_path, method, def_cat = None, None, None
        hparams = {}
        cmd = args[0] if args else STR_INFO

        if cmd in [STR_REMOVE, STR_INFO]:
            op_args = args[1:]
            hparams, used_args_ix = parse_hparams_args(op_args, defaults=False)

            k_fold_arg = re_in(r"(.+)-" + STR_FOLD, op_args)
            if k_fold_arg:
                method = k_fold_arg.group(0)
            elif STR_TEST in op_args:
                method = STR_TEST

            if method:
                used_args_ix.append(op_args.index(method))

            free_args_ix = subtract(range(len(op_args)), used_args_ix)

            if len(free_args_ix):
                data_path = op_args[free_args_ix[0]]
                if len(free_args_ix) > 1:
                    def_cat = op_args[free_args_ix[1]]
                elif len(free_args_ix) > 2:
                    Print.error(ERROR_WNAUA, raises=ArgsParseError)

        elif len(args) > 1:
            Print.error(ERROR_WNAUA)
            Print.warn(
                "Suggestion: evaluations %s" % cmd,
                raises=ArgsParseError
            )

        return cmd, data_path, method, def_cat, hparams

    def args_classify(self, args):
        """Parse classify arguments."""
        args = args.strip('"\'')
        if not args:
            Print.info(MSG_USER_INPUT_DOC)
            document = ''.join(sys.stdin.readlines())
            print("\n---")
        else:
            try:
                with open(args, "r", encoding=ENCODING) as fdoc:
                    document = fdoc.read()
            except IOError:
                Print.error(ERROR_NSF % args, raises=ArgsParseError)
        return document

    def args_live_test(self, args):
        """Parse live_test arguments."""
        args = split_args(args)
        path = None
        verbose = False
        folder_label = True
        path_required = False

        op_args = args
        op_args_ix = []

        folder_arg = intersect([STR_FILE, STR_FOLDER], op_args)
        if folder_arg:
            path_required = True
            folder_label = folder_arg[0] == STR_FOLDER
            op_args_ix.append(op_args.index(folder_arg[0]))

        if STR_VERBOSE in op_args:
            verbose = True
            op_args_ix.append(op_args.index(STR_VERBOSE))

        path_ix = subtract(range(len(op_args)), op_args_ix)
        if len(path_ix) == 1:
            path = op_args[path_ix[0]]
        elif len(path_ix) > 1:
            Print.error(
                ERROR_UA % op_args[path_ix[1]], raises=ArgsParseError
            )
        elif len(path_ix) == 0 and path_required:
            Print.error("A path must be given", raises=ArgsParseError)

        return path, folder_label, verbose

    def args_learn(self, args):
        """Parse learn arguments."""
        args = split_args(args)
        n_grams = 1
        doc_path = None

        if not args:
            Print.error(ERROR_WAN % (1, 0), raises=ArgsParseError)

        if args[0] not in CLF.get_categories():
            Print.error(ERROR_ICN % args[0], raises=ArgsParseError)

        op_args = args[1:]
        op_args_ix = []

        n_grams_arg = re_in(r"(.+)-" + STR_NGRAMS, op_args)
        if n_grams_arg:
            op_args_ix.append(op_args.index(n_grams_arg.group(0)))
            try:
                n_grams = int(n_grams_arg.group(1))
                if n_grams <= 0:
                    raise Exception
            except BaseException:
                Print.error(ERROR_WNGRAM, raises=ArgsParseError)

        def_cat_arg = subtract(range(len(op_args)), op_args_ix)
        if len(def_cat_arg) == 1:
            doc_path = op_args[def_cat_arg[0]]
        elif len(def_cat_arg) > 1:
            Print.error(ERROR_WNAUA, raises=ArgsParseError)

        if not doc_path:
            Print.info(MSG_USER_INPUT_DOC)
            document = ''.join(sys.stdin.readlines())
            print()
        else:
            try:
                with open(doc_path, "r", encoding=ENCODING) as fdoc:
                    document = fdoc.read()
            except IOError:
                Print.error(ERROR_NSF % doc_path, raises=ArgsParseError)

        return args[0], n_grams, document

    def args_save(self, args):
        """Parse save arguments."""
        args = split_args(args)

        # case 'model'
        if not args or args[0] == STR_MODEL:
            return STR_MODEL, None

        # case 'vocabulary'
        elif args[0] == STR_VOCABULARY:
            return STR_VOCABULARY, args[1] if len(args) == 2 else ''

        # case 'evaluations'
        elif args[0] == STR_EVALUATIONS:
            return STR_EVALUATIONS, None

        # case 'stopwords'
        elif args[0] == STR_STOPWORDS:
            threshold = None
            if len(args) > 1 and args[1]:
                try:
                    threshold = float(args[1])
                except ValueError:
                    Print.error(ERROR_WAT % "float", raises=ArgsParseError)

            return STR_STOPWORDS, threshold

        # otherwise
        else:
            Print.error(ERROR_UA % args[0], raises=ArgsParseError)

    def args_set(self, args):
        """Parse set arguments."""
        args = split_args(args)

        if not args:
            Print.error(ERROR_AR % 'set', raises=ArgsParseError)

        hparams, used_args_ix = parse_hparams_args(args)

        if len(used_args_ix) != len(args):
            Print.error(ERROR_WNAUA, raises=ArgsParseError)

        return hparams

    def preloop(self):
        """Hook method executed once when cmdloop() is called."""
        if readline and path.exists(HISTFILE):
            readline.read_history_file(HISTFILE)

    def precmd(self, line):
        """Hook method executed just before the command."""
        if line != 'EOF':
            line = line.lower()
        else:
            print('')
        return line

    def default(self, line):
        """Default error message."""
        Print.error('Unknown command: %s' % line)

    do_EOF = do_exit

    complete_get = complete_set
    complete_ld = complete_load
    complete_sv = complete_save
    complete_k_fold = complete_grid_search


def main():
    """Main function."""
    global MODELS
    prompt = SS3Prompt()
    prompt.prompt = '(pyss3) >>> '
    prompt.doc_header = "Documented commands (type help <command>):"

    Print.set_verbosity(VERBOSITY.VERBOSE)
    Print.info(
        'PySS3 Command Line v%s | Sergio Burdisso (sergio.burdisso@gmail.com).\n'
        'PySS3 comes with ABSOLUTELY NO WARRANTY. This is free software,\n'
        'and you are welcome to redistribute it under certain conditions\n'
        '(Type "license" for more details).\n'
        'Type "help" or "help <command>" for more information.\n'
        %
        (__version__),
        decorator=False
    )

    try:
        MODELS = [
            path.splitext(model_file)[0]
            for model_file in listdir(SS3.__models_folder__)
            if path.splitext(model_file)[1][1:] == STR_MODEL_EXT
        ]
        ARGS["load"] = MODELS
    except OSError:
        Print.warn("No local 'ss3_models' folder was found.")
        Print.warn(
            "Suggestion: either create a new model and train it or open this\n"
            "*             command line from a different folder "
            "(with models).\n"
        )

    try:
        prompt.cmdloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
        prompt.do_exit()


if __name__ == '__main__':
    main()
