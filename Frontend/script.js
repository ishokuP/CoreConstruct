function toggleFields(id) {
    var element = document.getElementById(id);
    if (element.style.display === "none" || element.style.display === "") {
        element.style.display = "block";
    } else {
        element.style.display = "none";
    }
}

function getFoundationPlan() {
    document.getElementById('foundation-plan').addEventListener('change', function (event) {
        const f_plan = event.target.files[0];

        if (f_plan) {
            console.log('File selected:', f_plan);
            document.getElementById("project-title").innerHTML = f_plan.name;
        }
    });
}

function renameFoundationPlan() {
    const f_plan = document.querySelector('input[type="file"]');
}