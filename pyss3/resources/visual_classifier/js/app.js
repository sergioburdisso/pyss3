"use strict";

var app = angular.module("ss3live", ["ngAnimate"]);

app.controller("mainCtrl", function($scope) {
  var max_v = 1;
  var chart = null;
  var active_cats = null;

  $scope.ss3 = null;
  $scope.loading = true;
  $scope.error = false;
  $scope.show_chart = false;
  $scope.word_info = false;
  $scope.scat = -1;
  $scope.ncats = 0;
  $scope.levels = [false, true, true];
  $scope.info = null;
  $scope.document = ''; 
  $scope.document_original = null;
  $scope.document_label = null;
  $scope.i_doc = -1;
  $scope.c_doc = null;
  $scope.f_doc = null;
  $scope.chart = {
    obj: null, // google chart object
    srow:  null, // selected row object (word,sent,par)
    level: 1, // 0 par, 1 sent, 2 word
    pars:  [],
    parsi: [],
    sents: [],
    sentsi:[],
    words: [],
    wordsi:[]
  }

  window.addEventListener("click", function(){
    __delayed_call__(function(){
      var $chart = $scope.chart;
      if (!("prev_srow" in this)){
        this.prev_srow = null;
      }
      if (this.prev_srow && this.prev_srow == $chart.srow){
        $chart.srow = null;
        $scope.word_info = false;
        if ($chart.obj)
          $chart.obj.setSelection({row:null,colum:null});
      }
      this.prev_srow = $chart.srow;
      $scope.$apply();
    });
  });

  $scope.keys = function(dict){
    return dict? Object.keys(dict) : [];
  }

  $scope.select_cat_menu = function(icat){
    if (icat != -1 && $scope.show_chart)
      $scope.chart.obj.setSelection({column: icat + 1});
    $scope.select_cat(icat);
  }

  $scope.update_document = function(){
    if ($scope.c_doc && $scope.document_original == $scope.document){
      $scope.f_doc = $scope.info.docs[$scope.c_doc]["file"][$scope.i_doc];
    }else{
      $scope.f_doc = null;
    }

  }

  $scope.get_clf_result = function(c, i){
    return $scope.info.categories[$scope.info.docs[c]["clf_result"][i]];
  }

  $scope.get_doc = function(c, i){
    $scope.loading = true;
    $scope.ss3 = null;
    $scope.i_doc = i;
    $scope.c_doc = c;
    $scope.f_doc = $scope.info.docs[c]["file"][i];
    $server.submit(
      'get_doc',
      $scope.info.docs[c]["path"][i],
      function(doc) {
        $scope.document = doc.content;
        $scope.document_original = doc.content;
        $scope.document_label = c;
        $scope.classify();
      }
    );
  }

  $scope.get_accuracy = function(c){
    var hits = $scope.info.docs[c]["file"].map(function(_, i_doc){
      return $scope.get_clf_result(c, i_doc) == c;
    });
    return sum(hits) / $scope.info.docs[c]["file"].length * 100;
  }

  $scope.get_cat_rgb = function(icat){
    return __rgb_colors__[icat];
  }

  $scope.get_cat_rgba = function(icat, alpha){
    return __rgba_colors__[icat] + alpha + ')';
  }

  $scope.get_window_height = function(){
    return document.body.clientHeight -  196 + 'px';
  }

  $scope.select_cat = function(icat, goto){
    $scope.scat = icat;
    $scope.word_info = false;
    if (goto)
      __goto__("hashtag");
  }

  $scope.edit = function(){
    $("#btn_edit").hide();
    __reset__();
    __delayed_call__(__update_textarea__);
    __goto__(0);
  }

  $scope.clear = function(){
    __reset__();
    $scope.document= "";
    __update_textarea__();
    __goto__(0);
  }

  $scope.new = function(){
    $scope.clear();
    $scope.edit();
  }

  $scope.get_cv_style = function(elem, ilevel){
    var style = {}
    var silevel = parseInt($scope.chart.level)
    var icat = $scope.scat;
    var n_levels = 1;  // $scope.levels.reduce((a, b) => a + b);
    var alpha = 0;

    if ($scope.chart.srow && ((!$scope.word_info && silevel == ilevel) || ($scope.word_info && ilevel == 2))){
      if (elem && elem.index == $scope.chart.srow.index){
        if (navigator.userAgent.toLowerCase().indexOf('chrome') > -1)
          style["border"] = "2px dotted #ee6e73";
        else
          style["border"] = "1px dotted #ee6e73";
      }else
        style["opacity"] = .3;
    }

    if (!elem || !$scope.levels[ilevel])
      return style;

    if (icat == -1){
      var acv = elem.cv.map(function(e, i){return active_cats.indexOf(i) != -1? e : -1});
      icat = elem.cv.indexOf(Math.max(...acv));
      // icat = elem.cv.indexOf(Math.max(...elem.cv));
    }

    if (ilevel == 2){ // if level == words
      alpha = elem.cv[icat];
      if (alpha > .005){
        style["text-decoration-line"] = "underline";
        style["text-decoration-color"] = alpha >= .01? $scope.get_cat_rgb(icat) : $scope.get_cat_rgba(icat, .4);
        style["text-decoration-style"] = "dotted";
      }
    }else{
      alpha = Math.min(elem.cv[icat], elem.wmv[icat]) / n_levels;
    }
    style['background-color'] = $scope.get_cat_rgba(icat, n_levels? alpha : 0);
    return style;
  }

  $scope.show_info = function(word){
    $scope.word_info = true;
    $scope.chart.srow = word;
    $scope.chart.srow.scat = $scope.scat != -1? $scope.scat : word.cv.indexOf(Math.max(...word.cv));
    if ($scope.chart.obj)
      $scope.chart.obj.setSelection({row:null,colum:null});
  }

  $scope.is_cat_active = function (cat_info) {
    return active_cats.indexOf(cat_info[0]) != -1;
  }

  $scope.on_chart_change = function(){
    var $chart = $scope.chart;
    if ($scope.show_chart){
      //TODO: this code below doesn't update
      // the view. Problems with Angular. 
      /*if ($chart.pars.length >= 10)
        $chart.level = 0;
      else if ($chart.sents.length > 6)
        $chart.level = 1;
      else
        $chart.level = 2;*/
      __create_chart__();
    }else{
      $("#chart").html("");
    }

    $chart.srow = null;
    $scope.word_info = false;
    /*if (!$scope.$$phase)
        $scope.$apply();*/
  }

  $scope.classify = function(){if (!$scope.document) return;
    $scope.error = "";
    $scope.loading = true;
    $scope.show_chart = false;
    __reset__();

    $server.submit(
      'classify',
      $scope.document,
      function(data, textStatus, jqXHR){
        if ("error" in data){
          $scope.error = data.error;
        }else{
          $scope.ss3 = data;
          $scope.scat = -1;
          active_cats = [];
          if (data.cvns.length == 2){
            active_cats.push(data.cvns[0][0]);
            active_cats.push(data.cvns[1][0]);
          }else{
            active_cats = __k_means_active_cat__(data.cvns);
          }
          __create_chart_values__();

          $("#btn_classify").hide();
          $("#btn_clear").hide();
          $("#btn_example").hide();
        }
        $scope.loading = false;
        $scope.word_info = false;

        $('#document').height("10px");
        M.textareaAutoResize($('#document')); // not working

        $scope.$apply();
      },
      function(jqXHR, textStatus, errorThrown){
        $scope.loading = false;
        $scope.error = STR_SERVER_ERROR;
        $scope.$apply();

        window.close();
      }
    );
  }

  function __k_means_active_cat__(cats){
    var cent = {neg: -1, pos: -1};  // centroids (one for each "group")
    var clust = {neg: [], pos: []};  // clusters (one for each "group")
    var new_cent_neg = cats[cats.length - 1][2];
    var new_cent_pos = cats[0][2];
    var active_cats = null;
    while (cent.pos != new_cent_pos || cent.neg != new_cent_neg){
      cent.neg = new_cent_neg;
      cent.pos = new_cent_pos;
      clust.neg = []; clust.pos = [];
      active_cats = [];
      for (var i=0, cat_cv; i < cats.length; i++){
        cat_cv = cats[i][2];
        if (Math.abs(cent.neg - cat_cv) < Math.abs(cent.pos - cat_cv)){
          clust.neg.push(cat_cv);
        }else{
          clust.pos.push(cat_cv);
          active_cats.push(cats[i][0]);
        }
      }
      new_cent_neg = clust.neg.reduce((a,b) => a + b, 0) / clust.neg.length;
      new_cent_pos = clust.pos.reduce((a,b) => a + b, 0) / clust.pos.length;
    }
    return active_cats;
  }

  function __create_chart_values__(){
    var $ss3 = $scope.ss3;
    var $chart = $scope.chart;
    var crow_pars = null;
    var crow_sents = null;
    var crow_words = null;
    var pari = 1;
    var senti = 1;
    var wordi = 1;

    $chart.pars = [];
    $chart.parsi = [];
    $chart.sents = [];
    $chart.sentsi = [];
    $chart.words = [];
    $chart.wordsi = [];
    for (var ip=0; ip < $ss3.pars.length; ip++){
      var par = $ss3.pars[ip];
      par.index = pari++;
      if (!(par.sents.length == 1 && par.sents[0].words.length == 1 && !par.sents[0].words[0].token)){
        if (crow_pars)
          crow_pars.add(par.cv);
        else
          crow_pars = par.cv.clone();
        $chart.pars.push(
          [par.sents[0].words[0].lexeme + "..."].concat(crow_pars)
        );
        $chart.parsi.push(par);
      }

      for (var is=0; is < par.sents.length; is++){
        var sent = par.sents[is];
        sent.index = senti++;

        if (sent.words.length > 1 || sent.words[0].token){
          if (crow_sents)
            crow_sents.add(sent.cv);
          else
            crow_sents = sent.cv.clone();
          $chart.sents.push(
            [sent.words[0].lexeme + "..."].concat(crow_sents)
          );
          $chart.sentsi.push(sent);
        }

        sent.dot = false;
        for (var iw=0; iw < sent.words.length; iw++){
          var word = sent.words[iw];
          word.index = wordi++;
          switch(word.token){
            case "":
              word.token = "<unkown>"; break;
            case "mnnbrr":
              word.token = "<money>"; break;
            case "nnbrrp":
              word.token = "<percentage>"; break;
            case "nnbrrd":
              word.token = "<date>"; break;
            case "nnbrrt":
              word.token = "<temp>"; break;
            default:
              word.token = word.token.replace("nnbrr", "<number>");
          }

          if (word.token && word.token != "<unkown>"){
            if (crow_words)
              crow_words.add(word.cv);
            else
              crow_words = word.cv.clone();
            $chart.words.push(
              [word.lexeme].concat(crow_words)
            );
            $chart.wordsi.push(word);
          }
          word.lexeme = __split_lexeme__(word.lexeme);

          if (!sent.dot && word.lexeme[1])
            sent.dot = true;
        }
      }
    }
  }

  function __create_chart__(){
    var data = new google.visualization.DataTable();
    var $chart = $scope.chart;
    var $ss3 = $scope.ss3;
    var rows = null;
    var xcolum = "";
    var options = {
      // theme: 'maximized',
      theme: 'material',
      colors: __rgb_colors__,
      lineWidth: 4,
      vAxis: {
        title: 'cumulated confidence value (cv)',
        textPosition: 'out'
      },
      hAxis: {
        title: '',
        textPosition: 'out'
      },
      legend: { position: 'none' },
      pointSize: 8,
      width: '100%',
      height: 500,
      chartArea:{
        top: 10,
        height: 450
      }
    };
    $chart.obj = new google.visualization.LineChart(document.getElementById('chart'));

    switch(parseInt($chart.level)){
      case 0:
        xcolum = 'paragraph';
        rows = $chart.pars;
        break;
      case 1:
        xcolum = 'sentence';
        rows = $chart.sents;
        break;
      case 2:
        xcolum = 'word';
        rows = $chart.words;
        break;
    }

    options.hAxis.title = xcolum;
    data.addColumn('string', xcolum);
    for (var icat=0; icat < $ss3.ci.length; icat++){
      data.addColumn('number', $ss3.ci[icat]);
    }
    data.addRows(rows);

    function selectHandler() {
      var item = $chart.obj.getSelection()[0];
      var $id = null;
      if (item) {
        $scope.select_cat(item.column-1);

        if (item.row != null){
          switch(parseInt($chart.level)){
            case 0: $chart.srow = $chart.parsi[item.row]; $id = "p-";
              break;
            case 1: $chart.srow = $chart.sentsi[item.row]; $id = "s-";
              break;
            case 2: $chart.srow = $chart.wordsi[item.row]; $id = "w-";
              break;
          }
          $scope.word_info = false;
          __delayed_call__(function(){if ($chart.srow) __goto__($id + $chart.srow.index, 500, -50);})

        }
      }

      if (!item || item.row == null){
        $chart.srow = null;
      }

      if (!$scope.$$phase)
        $scope.$apply();
    }

    $chart.obj.draw(data, options);
    google.visualization.events.addListener($chart.obj, 'select', selectHandler);
  }

  function __reset__(){
    $scope.error = "";
    $scope.ss3 = null;
    $scope.show_chart = false;
    $scope.on_chart_change();
  }

  $server.submit("get_info", '', function(data){
    $scope.info = data;
    var cats = data.categories.slice(0, -1);
    $scope.ncats = cats.length;
    __rgba_colors__ = cats.map(function(_, i){
      return "rgba(" + __get_rgb__(i, cats.length) + ',';
    });
    __rgb_colors__ = cats.map(function(_, i){
      return "rgb(" + __get_rgb__(i, cats.length) + ')';
    });
    $scope.loading = false;
    $scope.$apply();
    $(document).ready(function(){
      $('.collapsible').collapsible();
    });
  });

  window.onfocus = function(){
    $server.submit(
      "ack", '', undefined, function(){
        $scope.error = STR_SERVER_ERROR;
        $scope.$apply();
        window.close();
      });
  };
});

function __split_lexeme__(lexeme) {
  var re = /^[^\w\d]+/.exec(lexeme);
  var re_word = /[\w\dÀ-ÿ]+((\s|-)?[\w\dÀ-ÿ]+)*/;
  var word = null;
  var output = ["", "", ""];

  if (lexeme.indexOf(" -") != -1)
    lexeme = lexeme.replace(" -", "-");

  if (!lexeme || !re_word.exec(lexeme)){
    output[0] = lexeme;
    return output;
  }
  
  if (re)
      output[0] = lexeme.substr(0, re[0].length);

  word = re_word.exec(lexeme);
  output[1] = word[0];
  output[2] = lexeme.substr(word.index + word[0].length);

  return output;
}

function __update_textarea__(){
  __delayed_call__(function(){M.textareaAutoResize($('#document'));});
  $('#document').height("10px");
  $('#document').focus();
}
