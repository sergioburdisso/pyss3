"use strict";

var STR_SERVER_ERROR = "Ups! couldn't connect to the SS3 Server :(";
var $server = {};
$server.url = '';
$server.submit = function(path, data, success, error){
      $.ajax({
      url: $server.url + path,
      data: encodeURIComponent(data),
      processData: false,
      contentType: false,
      type: 'POST',
      success: success,
      error: error
    });
};

var __start_space__ = Array('(', '[', '"', "¡", "¿", "#");

var __no_start_space__ = Array('(', '[', '"', "'", "¡", "¿", "-", "#", "/");

var __rgba_colors__ = [];

var __rgb_colors__ = [];

function __get_rgb__(i, T){
  var binary = ["255,64,129", "0,176,255"/*"0,230,118"*/];
  var multiclass = [
    "236,64,122",  // pink lighten-1
    "220,231,117", // lime lighten-2
    "139,195,74",  // light-green
    "206,147,216", // purple lighten-3
    "77,182,172",  // teal lighten-2
    "255,193,7",   // amber
    "33,150,243",  // blue
    "255,138,101", // deep-orange lighten-2
  ];
  if (T == 2){
    return binary[i];
  }else{
    if (i < multiclass.length){
      return multiclass[i];
    }else{
      return __new_rgb__(T, i);
    }
  }
}

function __new_rgb__(T, i) {
    var r, g, b;
    var h = i / T;
    var i = (h * 6)|0;
    var f = h * 6 - i;
    switch(i % 6){
        case 0: r = 1;     g = f;     b = 0; break;
        case 1: r = 1 - f; g = 1;     b = 0; break;
        case 2: r = 0;     g = 1;     b = f; break;
        case 3: r = 0;     g = 1 - f; b = 1; break;
        case 4: r = f;     g = 0;     b = 1; break;
        case 5: r = 1;     g = 0;     b = 1 - f; break;
    }
    return ((r * 255)|0) + ',' + ((g * 255)|0) + ',' + ((b * 255)|0);
}

function __delayed_call__(func, time){
  time = time || 100;
  setTimeout(func, time);
}

function __goto__(top, time, offset){
  offset = offset || 0;
  if (top && Number.isNaN(Number(top)) && top.constructor == String)
    top = $('#' + top).offset().top;

  if (time === undefined)
    time = 700;

  $('html, body').animate({"scrollTop": top + offset}, time, 'easeOutExpo');
}

function sum(arr){
  var res = 0, n = arr.length;
  while(n--) {
    res += +arr[n];
  }
  return res;
}

Array.prototype.add = function(v) {
  for(var i=0; i < v.length; i++) this[i] += v[i];
}

Array.prototype.clone = function(){return this.slice(0);}
