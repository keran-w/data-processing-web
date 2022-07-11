$(document).ready(function () {

    $("[id='div_id_Target Column']").addClass("formpage1").css({
        "width": "1220px",
        "display": "inline-block"
    });

    $("[id^='div_id_var-']").addClass("formpage1");

    $("[class^='form-label']").css("font-size", "24px");
    $("[class='form-check']").css("display", "inline-block").removeClass('form-check');
    $("[class='form-check-input']").addClass("btn-check").removeClass('form-check-input');
    $("[class='form-check-label']").addClass("btn btn-outline-dark").removeClass('form-check-label').css({
        "width": "400px",
        'margin': "5px 0 5px 0"
    });

    $("[id^='id_var-']").next('label').css("width", "240px");
    $("[id^='id_Train Methods']").next('label').css("margin", "15px 0 15px 0");

    $(':input').filter(function(){return this.value=='delete'}).next('label').addClass("btn-outline-danger").removeClass('btn-outline-dark');

    $("[id='div_id_Impute Methods']").addClass("formpage2");
    $("[id='div_id_Sampling Methods']").addClass("formpage2").css('margin-top', '80px');
    $("[id='div_id_Selection Methods']").addClass("formpage2").css('margin-top', '80px');

    $("[id='div_id_Train Methods']").addClass("formpage3").css('margin-top', '100px');

    $(".formpage2,.formpage3").hide();

    $("#btnpage2b").click(function () {
        $(".formpage1").show();
        $(".formpage2,.formpage3").hide();
    });

    $("#btnpage1n,#btnpage3b").click(function () {
        $(".formpage2").show();
        $(".formpage1,.formpage3").hide();
    });

    $("#btnpage2n").click(function () {
        $(".formpage3").show();
        $(".formpage1,.formpage2").hide();
    });
});