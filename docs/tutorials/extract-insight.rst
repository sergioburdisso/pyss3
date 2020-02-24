.. _extract-insight:

******************************************************************
Getting the text fragments involved in the classification decision
******************************************************************

.. raw:: html

    <br>
    <div style="text-align:right; color: #585858"><i>To <b>open and run</b> this notebook <b style="color:#E66581">online</b>, click here: <a href="https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples/extract_insight.ipynb" target="_blank"><img src="https://mybinder.org/badge_logo.svg" style="display: inline"></a></i></div>
    <br>
    <br>

In `this notebook <https://github.com/sergioburdisso/pyss3/blob/master/examples/extract_insight.ipynb>`__, we will see how we can use the
`PySS3 <https://github.com/sergioburdisso/pyss3>`__ Python package to
ask the text classifier not only to classify a document but also to give
us the ``list`` of text fragments its classification decision was based
on.

Let us begin! First, we need to import the modules we will be using:

.. code:: python

    from pyss3 import SS3
    from pyss3.util import Dataset

Then, before moving any further, we will unzip the training data. Since
it is located in `the same
directory <https://github.com/sergioburdisso/pyss3/tree/master/examples>`__
as this notebook file
(`extract\_insight.ipynb <https://github.com/sergioburdisso/pyss3/blob/master/examples/extract_insight.ipynb>`__),
we could simply use the following command-line command:

.. code:: shell

    !unzip -u datasets/topic.zip -d datasets/


Let's create a new instance of the SS3 classifier. We're going to use
the same dataset that is used in the `Topic
Categorization <https://pyss3.readthedocs.io/en/latest/tutorials/topic-categorization.html#topic-categorization>`__
tutorial. This dataset was created collecting tweets with hashtags
related to these 8 different categories: *“art&photography”,
“beauty&fashion”, “business&finance”, “food”, “health”, “music”,
“science&technology” and “sports”*.

.. code:: python

    # [create a new instance of the SS3 classifier]
    # Just ignore those hyperparameter values (s=0.32, l=1.24, p=1.1)
    # they were obtained from the tutorial (after performing hyperparameter optimization)
    # We could've been used just the default values simply with
    # clf = SS3()
    # but classification results would have been suboptimal (not optimized)
    clf = SS3(s=0.32, l=1.24, p=1.1)
    
    # Let's load the training set
    x_train, y_train = Dataset.load_from_files("datasets/topic/train", folder_label=False)
    
    # Let the training begin...
    clf.train(x_train, y_train, n_grams=3)


.. parsed-literal::

     Training: 100%|██████████| 8/8 [00:36<00:00,  4.57s/it]


We will use the following example document for SS3 to give us the text
parts involved in classifying it:

    .. rubric:: Effects of intensive short-term dynamic psychotherapy on
       social cognition in major depression
       :name: effects-of-intensive-short-term-dynamic-psychotherapy-on-social-cognition-in-major-depression

    Background: Social cognition is commonly affected in psychiatric
    disorders and is a determinant of quality of life. However, there
    are few studies of treatment.

    Objective: To investigate the efficacy of intensive short-term
    dynamic psychotherapy on social cognition in major depression.

    Method: This study used a parallel randomized control group design
    to compare pre-test and post-test social cognition scores between
    depressed participants receiving ISTDP and those allocated to a
    wait-list control group. Participants were adults (19–40 years of
    age) who were diagnosed with depression. We recruited 32
    individuals, with 16 participants allocated to the ISTDP and control
    groups, respectively. Both groups were similar in terms of age, sex
    and educational level.

    Results: Multivariate analysis of variance (MANOVA) demonstrated
    that the intervention was effective in terms of the total score of
    social cognition: the experimental group had a significant increase
    in the post-test compared to the control group. In addition, the
    experimental group showed a significant reduction in the negative
    subjective score compared to the control group as well as an
    improvement in response to positive neutral and negative states.
    Conclusion: Depressed patients receiving ISTDP show a significant
    improvement in social cognition post treatment compared to a
    wait-list control group.

We will assign it to the ``document`` variable:

.. code:: python

    document="""
    Effects of intensive short-term dynamic psychotherapy on social cognition in major depression
    ---
    
    Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality of life. However, there are few studies of treatment.
    Objective: To investigate the efficacy of intensive short-term dynamic psychotherapy on social cognition in major depression.
    Method: This study used a parallel randomized control group design to compare pre-test and post-test social cognition scores between depressed participants receiving ISTDP and those allocated to a wait-list control group. Participants were adults (19–40 years of age) who were diagnosed with depression. We recruited 32 individuals, with 16 participants allocated to the ISTDP and control groups, respectively. Both groups were similar in terms of age, sex and educational level.
    Results: Multivariate analysis of variance (MANOVA) demonstrated that the intervention was effective in terms of the total score of social cognition: the experimental group had a significant increase in the post-test compared to the control group. In addition, the experimental group showed a significant reduction in the negative subjective score compared to the control group as well as an improvement in response to positive neutral and negative states.
    Conclusion: Depressed patients receiving ISTDP show a significant improvement in social cognition post treatment compared to a wait-list control group.
    """

Now, before we ask SS3 to extract those fragments that were relevant for
classifying this document, we will ask SS3 to classify it.

.. code:: python

    clf.classify_label(document)




.. parsed-literal::

    'health'



Among the 8 learned category labels, SS3 decided to assign the label
``'health'`` to it, which we, as humans, can tell it is the correct
decision.

Now we are ready to ask SS3 to extract the relevant fragments for us. To
do this, we will use the ``clf.extract_insight()`` method. This 
method, given a document, returns the pieces of text that were involved
in the classification decision, along with the *confidence values*
associated with each (Its documentation is available
`here <https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.extract_insight>`__).

.. code:: python

    fragments = clf.extract_insight(document)
    
    print("How many text fragments were extracted?", len(fragments))


.. parsed-literal::

    How many text fragments were extracted? 17


Let's see what the first fragment looks like...

.. code:: python

    fragments[0]




.. parsed-literal::

    ('Effects of intensive short-term dynamic psychotherapy on social cognition in major depression',
     0.6793249876085043)



As we can see, each returned fragment is a pair of the form
``(text fragment, confidence value)``, and therefore, if we want only
the text fragment we can select only the first component:

.. code:: python

    print("Text:", fragments[0][0])
    print()
    print("Confidence value:", fragments[0][1])


.. parsed-literal::

    Text: Effects of intensive short-term dynamic psychotherapy on social cognition in major depression
    
    Confidence value: 0.6793249876085043


Now, let's take a look at the entire ``fragments`` list:

.. code:: python

    fragments




.. parsed-literal::

    [('Effects of intensive short-term dynamic psychotherapy on social cognition in major depression',
      0.6793249876085043),
     ('Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality of life. However, there are few ',
      0.6106375494375872),
     ('age) who were diagnosed with depression. We recruited 32 individuals, with 16 participants allocated to the ISTDP ',
      0.5296214999281954),
     ('of variance (MANOVA) demonstrated that the intervention was effective in terms of the total score of social cognition: the experimental group had ',
      0.5290513100358483),
     ('Objective: To investigate the efficacy of intensive short-term dynamic psychotherapy on social cognition in major depression.',
      0.3918906766905612),
     ('group showed a significant reduction in the negative subjective score compared to the control group as ',
      0.2982249447945927),
     ('group had a significant increase in the post-test compared to the control group',
      0.28696339456321973),
     ('there are few studies of treatment.', 0.28538479404883194),
     ('Method: This study used a parallel randomized ', 0.2600748912571276),
     ('in response to positive neutral and negative states.', 0.24862272509122232),
     ('improvement in social cognition post treatment compared to a wait-list control group',
      0.23100026122643016),
     ('Conclusion: Depressed patients receiving ISTDP show a significant improvement in social ',
      0.21682403869685085),
     (' Participants were adults (19–40 years of ', 0.11733643643026903),
     ('post-test social cognition scores between depressed participants receiving ISTDP ',
      0.06366606387267651),
     ('ISTDP and those allocated to a wait', 0.030070886155898154),
     ('Both groups were similar in terms of age, sex and ', 0.025867692840869892),
     ('group design to compare pre-test and ', 0.018493850304321317)]



As we can see, fragments are returned in a ``list`` that is ordered by
confidence value, which is great, the further away a fragment is from
the first one, the less confidence SS3 has that is relevant to the
assigned category. This is really desirable since in "real life"
documents will be arbitrarily long, we can always use the top ``n``
elements, for example, let's select the top 3 elements:

.. code:: python

    fragments[:3]




.. parsed-literal::

    [('Effects of intensive short-term dynamic psychotherapy on social cognition in major depression',
      0.6793249876085043),
     ('Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality of life. However, there are few ',
      0.6106375494375872),
     ('age) who were diagnosed with depression. We recruited 32 individuals, with 16 participants allocated to the ISTDP ',
      0.5296214999281954)]



And that's all! is it? want to go a little bit deeper? the following
section will show some more advanced features the ``extract_insight``
method has, just in case some of them can be useful to you.

--------------

What about the other categories?
================================

SS3 provides a version of the ``clf.classify_label``
method for `multi-label
classification <https://en.wikipedia.org/wiki/Multi-label_classification>`__,
it is called ``classify_multilabel``. So let's ask SS3 to try to
classify again the document, but this time getting rid of the
"select-only-one-category" constraint imposed by ``classify_label``.

.. code:: python

    clf.classify_multilabel(document)




.. parsed-literal::

    ['health', 'science&technology']



Among the 8 learned category labels, this time, SS3 decided to assign
not only the ``'health'`` label but also ``science&technology`` too,
which we, as humans, again can tell that both are correct since the
document is clearly a scientific article related to health.

The problem is that, if we use ``extract_insight`` again in the same
way, it will obviously show us the same result, that is, the fragments
related to ``'health'`` (the category assigned if it had to select only
one), how do we tell SS3 we want to extract fragments related to other
categories? use the ``cat`` argument!

For instance, if we want SS3 to give us the text fragments that were
used for classifying the document as ``science&technology``, we can do
as follows:

.. code:: python

    fragments = clf.extract_insight(document, cat="science&technology")
    
    fragments[:3]




.. parsed-literal::

    [('Method: This study used a parallel randomized control group design to compare pre-test and post',
      0.5495270398208789),
     ('Objective: To investigate the efficacy of intensive short-term dynamic psychotherapy on social cognition in major depression.',
      0.4810320116282637),
     ('Conclusion: Depressed patients receiving ISTDP show a significant improvement in social cognition post treatment compared to a wait-list control group.',
      0.4397448233815649)]



we can see that, unlike the previous ones, these fragments focus less on
health-related aspects and much more on science/scientific ones, SS3
even gave us the Method, Objective and Conclusion well-known sections of
research papers. For instance, if we read the first fragment without any
context, "Method: This study used a parallel randomized control group
design to compare pre-test and post", we as humans, can clearly see it
is related to science.

Just for fun, let's force SS3 to extract the text fragments that he
would use to classify the document, in a parallel universe, as
``sports``-ish.

.. code:: python

    fragments = clf.extract_insight(document, cat="sports")
    
    fragments[:3]




.. parsed-literal::

    [('the negative subjective score compared to the control group as ',
      0.08070207011696581),
     ('of the total score of social cognition: ', 0.06487662686978977),
     ('-test social cognition scores between depressed participants ',
      0.04261894232918068)]



We can see a pattern here, namely, fragments are talking about scores,
which again is the logical answer.

--------------

How to control the size of the fragments?
=========================================

*TL;DR:* Use the ``window_size`` argument!

If not given, by default ``window_size=3``, but bigger values produce
longer fragments while smaller, you guessed it! shorter ones. Let's try
out some values.

.. code:: python

    fragments = clf.extract_insight(document, window_size=0) # window_size=0
    
    fragments[:3]




.. parsed-literal::

    [('Effects of ', 0.34410723095944096),
     ('total ', 0.32683582484809587),
     ('psychiatric ', 0.2860576039598297)]



.. code:: python

    fragments = clf.extract_insight(document, window_size=1) # window_size=1
    
    fragments[:3]




.. parsed-literal::

    [('were diagnosed with depression. We ', 0.47386514201385327),
     ('Effects of intensive short', 0.3881150202849344),
     ('the total score ', 0.3268857739319143)]



.. code:: python

    fragments = clf.extract_insight(document, window_size=2) # window_size=2
    
    fragments[:3]




.. parsed-literal::

    [('Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality ',
      0.6041370831998978),
     ('who were diagnosed with depression. We recruited 32 individuals, with ',
      0.49028660933765983),
     ('Effects of intensive short-term dynamic psychotherapy on ',
      0.45190110601897143)]



.. code:: python

    fragments = clf.extract_insight(document, window_size=5) # window_size=5
    
    fragments[:3]




.. parsed-literal::

    [('Multivariate analysis of variance (MANOVA) demonstrated that the intervention was effective in terms of the total score of social cognition: the experimental group had a significant increase in the post-test compared to the control group. In addition, the experimental group showed a significant reduction in the negative subjective score compared to the control group as well as an improvement in response to positive neutral and negative states.',
      1.369701510164149),
     ('Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality of life. However, there are few studies of treatment.',
      0.8960223434864192),
     ('Effects of intensive short-term dynamic psychotherapy on social cognition in major depression',
      0.6793249876085043)]



Nice, it works like a charm! but... **what if I want the size of the
fragments to be exactly one paragraph each? or... one sentence each?**
Instead of ``window_size``, use the ``level`` argument! this argument
takes exactly 3 possible values: ``'paragraph'``, ``'sentence'``, or the
default ``'word'``, which is used when the ``level`` argument is not
given. This argument tells SS3 the "level" at which fragments are to be
constructed.

For instance, let's ask SS3 to give us the most relevant paragraph that
was used for classifying the document as scientific:

.. code:: python

    fragments = clf.extract_insight(document, cat="science&technology", level="paragraph")
    
    print("The coolest paragraph is:\n\n", fragments[0][0])
    print()
    print("And its confidence value:", fragments[0][1])


.. parsed-literal::

    The coolest paragraph is:
    
     Method: This study used a parallel randomized control group design to compare pre-test and post-test social cognition scores between depressed participants receiving ISTDP and those allocated to a wait-list control group. Participants were adults (19–40 years of age) who were diagnosed with depression. We recruited 32 individuals, with 16 participants allocated to the ISTDP and control groups, respectively. Both groups were similar in terms of age, sex and educational level.
    
    And its confidence value: 1.4044308397641223


And what about the 3 most relevant sentences to ``'health'``?

.. code:: python

    fragments = clf.extract_insight(document, level="sentence")
    
    fragments[:3]




.. parsed-literal::

    [('Results: Multivariate analysis of variance (MANOVA) demonstrated that the intervention was effective in terms of the total score of social cognition: the experimental group had a significant increase in the post-test compared to the control group',
      0.8216551616603024),
     ('Effects of intensive short-term dynamic psychotherapy on social cognition in major depression',
      0.6793249876085043),
     ('Background: Social cognition is commonly affected in psychiatric disorders and is a determinant of quality of life',
      0.6041370831998978)]

Cool! however, what if I want to redefine what a paragraph, sentence or
a word is considered to be for SS3?... well, what? OK... I guess your
working with a different type of text, that is, a text that for some
reason has a special format.

Let's now suppose we are working with "weird" documents in which biggest
blocks are delimited by the @ character (as if they were paragraph), and
these "@-paragraph" blocks are, in turn, composed of smaller blocks
delimited by the # character (as if they were sentences). Let's also
suppose that we want to analyze the following document:

.. code:: python

    weird_document="@Effects of#intensive short-term dynamic psychotherapy@on social cognition#in major depression@"

As we can see, this weird document has two "@-paragraphs" with two
"#-sentences" each, if we use the ``extract_insight`` method as before,
it will only return a single fragment since SS3 sees this weird document
as a "normal" one, a document with a single paragraph with a single
sentence:

.. code:: python

    fragments = clf.extract_insight(weird_document, level="sentence")
    
    fragments




.. parsed-literal::

    [('@Effects of#intensive short-term dynamic psychotherapy@on social cognition#in major depression@',
      0.6793249876085043)]



Therefore, we need to tell SS3 that we want to redefine these concepts
so that "he" can be aware of those "@-paragraphs" and "#-sentences", we
can do this by using the ``set_block_delimiters`` method (documentation
`here <https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.set_block_delimiters>`__),
as follows:

.. code:: python

    clf.set_block_delimiters(parag="@", sent="#")

Now, let's try again...

.. code:: python

    fragments = clf.extract_insight(weird_document, level="sentence")
    
    fragments




.. parsed-literal::

    [('Effects of', 0.34410723095944096),
     ('in major depression', 0.2021045058091867),
     ('intensive short-term dynamic psychotherapy', 0.10779387505953043),
     ('on social cognition', 0.025319375780346178)]



Perfect! this time, all four "#-sentences" got caught :)

Let's see what happens with the @-paragraphs:

.. code:: python

    fragments = clf.extract_insight(weird_document, level="paragraph")
    
    # ignore this line, just restoring the default delimiter values
    # just in case you want to re-run some of the code given previously
    # with the "normal document" (not the @weirdo# one)
    clf.set_block_delimiters(parag="\n", sent="\.")
    
    fragments




.. parsed-literal::

    [('Effects of#intensive short-term dynamic psychotherapy', 0.4519011060189714),
     ('on social cognition#in major depression', 0.2274238815895329)]



As expected, it worked like a charm :D .... but... what if.. just jokin'
no more buts (for now).

--------------

Just remember that all these last sections addressed more "advanced"
cases, most of the time you should be just fine with plain
``clf.extract_insight(document)`` and simply using different
``window_size`` values when needed.

.. raw:: html

    BTW, wow! you've reached this far! you deserve a nice coffee, don't you? <img src="https://github.githubassets.com/images/icons/emoji/unicode/2615.png" style="margin-bottom: 0; display: inline" width="20"><img src="https://github.githubassets.com/images/icons/emoji/unicode/1f609.png" style="margin-bottom: 0; display: inline" width="20"> Have an awesome day.
