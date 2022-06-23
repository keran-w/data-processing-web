// Method to upload a valid excel file
function upload() {
    var files = document.getElementById('file_upload').files;
    if (files.length == 0) {
        alert("Please choose any file...");
        return;
    }
    var filename = files[0].name;
    var extension = filename.substring(filename.lastIndexOf(".")).toUpperCase();
    if (extension == '.XLS' || extension == '.XLSX') {
        //Here calling another method to read excel file into json
        excelFileToJSON(files[0]);
    }
    else {
        alert("Please select a valid excel file.");
    }
    document.getElementById("upload_form").style.display = "none";
}
//Method to read excel file and convert it into JSON 
function excelFileToJSON(file) {
    try {
        var reader = new FileReader();
        reader.readAsBinaryString(file);
        reader.onload = function (e) {
            var data = e.target.result;
            var workbook = XLSX.read(data, {
                type: 'binary',
                sheetRows: 16
            }
            );
            var result = {
            };
            var firstSheetName = workbook.SheetNames[0];
            //reading only first sheet data
            var jsonData = XLSX.utils.sheet_to_json(workbook.Sheets[firstSheetName]);
            //displaying the json result into HTML table
            displayJsonToHtmlTable(jsonData);
        }
    }
    catch (e) {
        console.error(e);
    }
}
//Method to display the data in HTML Table
function displayJsonToHtmlTable(jsonData) {
    var table = document.getElementById("display_excel_data");
    if (jsonData.length > 0) {
        var htmlData = '<tr><th>Student Name</th><th>Address</th><th>Email ID</th><th>Age</th></tr>';
        for (var i = 0; i < jsonData.length; i++) {
            var row = jsonData[i];
            htmlData += '<tr><td>' + row["Student Name"] + '</td><td>' + row["Address"]
                + '</td><td>' + row["Email ID"] + '</td><td>' + row["Age"] + '</td></tr>';
        }
        table.innerHTML = htmlData;
    }
    else {
        table.innerHTML = 'There is no data in Excel';
    }
}