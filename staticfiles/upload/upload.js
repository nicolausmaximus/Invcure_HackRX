// select the drop area element
var $form = $('.box');
// self-invoking function
// detect the drag&drop and other features
var isAdvancedUpload = function() {
  var div = document.createElement('div');
  return (('draggable' in div) || ('ondragstart' in div && 'ondrop' in div)) && 'FormData' in window && 'FileReader' in window;
}();

// show the selected file
// get the document element
var $input    = $form.find('input[type="file"]'),
    $label    = $form.find('label'),
    $boxInput = $form.find('.box__input'),
    // change the label of the choose file button
    showFiles = function(files) {
      //$label.text(files.length > 1 ? ($input.attr('data-multiple-caption') || '').replace( '{count}', files.length ) : $label.html('<u>' + files[ 0 ].name + '</u>'));
      files.length > 1 ? ($input.attr('data-multiple-caption') || '').replace( '{count}', files.length ) : $label.html('<u>' + files[ 0 ].name + '  ' +' <i class="fa fa-minus-circle" aria-hidden="true"></i>' +'</u>');
    };

// add style if feature is supported
if (isAdvancedUpload) {
  $form.addClass('has-advanced-upload');

  var droppedFiles = false;
  // listen to the drag events
  $form.on('drag dragstart dragend dragover dragenter dragleave drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
  })
  .on('dragover dragenter', function() {
    $form.addClass('is-dragover');
  })
  .on('dragleave dragend drop', function() {
    $form.removeClass('is-dragover');
  })
  .on('drop', function(e) {
    droppedFiles = e.originalEvent.dataTransfer.files; // file dropped
    showFiles(droppedFiles);
    // console.log(droppedFiles);

    // read dropped file
    var file = droppedFiles[0];

    var reader = new FileReader();
    // listen to the reader's load event
    reader.onload = function (evt){
      //jsonObj = JSON.parse(evt.target.result);
      console.log(evt.target.result);
      
    }
    reader.readAsText(file);
  });

  // submit data
  // e.preventDefault();
  //
  // var ajaxData = new FormData($form.get(0)); // get data from the Form
  //
  // if (droppedFiles){
  //   // loop over dropped files
  //   $.each( droppedFiles, function(i, file){
  //     ajaxData.append( $input.attr('name'), file);
  //   })
  // }

  // $.ajax({
  //   // upload and ajax method goes here
  // });
}

// upload using Ajax

// when select file with input control
// show the selected file name
$input.on('change', function(e) {
  showFiles(e.target.files);
});