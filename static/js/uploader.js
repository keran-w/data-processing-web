function show_sections(i1, i2, i3, i4, i5) {
    var all_section_ids = ['1', '2', '3', '4', '5'];

    var arr = [i1, i2, i3, i4, i5];
    var show_ids = arr.filter((x) => Boolean(x));
    var hide_ids = all_section_ids.filter((el) => !show_ids.includes(el));

    for (let i = 0; i < show_ids.length; i++) {
        document.getElementById("section-" + show_ids[i]).style.display = "block";
    }

    for (let i = 0; i < hide_ids.length; i++) {
        document.getElementById("section-" + hide_ids[i]).style.display = "none";
    }
}