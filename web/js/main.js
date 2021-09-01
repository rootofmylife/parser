$(document).on("change", ".custom-file-input",
  function (event) {
    $(this).next(".custom-file-label").html(event.target.files[0].name);
  }
);

$(document).ready(function () {

  var deletemode = false;
  $(".actmode").click(function () {
    var this_el = $(this);
    var this_id = this_el.data("id");
    console.log(deletemode, this_id);
    if (!deletemode) {
      window.location.href = "/annotatrix?treebank_id=" + this_id;
    }
  });

  $("#actionswitch").change(function () {
    if (this.checked) {
      $(".actmode").html("Delete");
      console.log("checked");
      deletemode = true;
    } else {
      $(".actmode").html("Edit");
      console.log("unchecked");
      deletemode = false;
    }
  });
});