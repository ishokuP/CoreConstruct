function toggleFields(id, arrowId) {
    var element = document.getElementById(id);
    var arrow = document.getElementById(arrowId);

    if (element.style.display === "none" || element.style.display === "") {
        element.style.display = "block";
        arrow.classList.add("down");
    } else {
        element.style.display = "none";
        arrow.classList.remove("down");
    }
}

let selectedFile = null;

function getFoundationPlan() {
    document.getElementById('foundation-plan').addEventListener('change', function (event) {
        selectedFile = event.target.files[0];

        if (selectedFile) {
            console.log('File selected:', selectedFile);
            const fileNameWithoutExtension = selectedFile.name.replace(/\.[^/.]+$/, "");
            document.getElementById("project-title").innerText = fileNameWithoutExtension;
        }
    });
}

function saveFoundationPlan() {
    // Send a request to save the RLFoundationPadded.json as a .cct file
    fetch('/save-foundation-plan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fileType: 'cct', fileName: 'foundation-plan.cct' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            alert(data.message);  // Success message
            window.location.href = `/download/${data.fileName}`;  // Trigger download
        } else {
            alert("Error: " + data.error);  // Error message
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to save the file.');
    });
}





function saveAsFoundationPlan() {
    document.getElementById('saveAsModal').style.display = 'flex';
}

function closeModal() {
    document.getElementById('saveAsModal').style.display = 'none';
}

function confirmSaveAs() {
    const userFileName = document.getElementById("project-title").innerText.trim();
    const fileType = document.getElementById("file-type-selector").value;
    let newFileName = userFileName ? userFileName : 'renamed-foundation-plan';

    // Set the file extension based on selected format
    switch (fileType) {
        case 'cct':
            newFileName += '.cct';  // Default behavior for .cct (same as Save button)
            break;
        case 'png':
            newFileName += '.png';  // .png selection triggers the image save
            break;
        default:
            newFileName += '.cct';  // Default to .cct if nothing is selected
            break;
    }

    // Send the selected file type and file name to the backend
    fetch('/save-as-foundation-plan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fileType: fileType, fileName: newFileName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            alert(data.message);  // Success message
            window.location.href = `/download/${newFileName}`;  // Trigger download
        } else {
            alert("Error: " + data.error);  // Error message
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to save the file.');
    });

    closeModal();  // Close the modal after confirming
}


document.querySelectorAll(".select-group").forEach(function (dropdownGroup) {
    const button = dropdownGroup.querySelector("button");
    const selectLabel = dropdownGroup.querySelector('#select-label');
    const dropdown = dropdownGroup.querySelector(".dropdown");
    const options = dropdownGroup.querySelectorAll(".option");
    const arrow = button.querySelector(".arrow");

    let selectedOption = null;

    button.addEventListener("click", function (e) {
        e.preventDefault();
        toggleDropdown(dropdown, arrow);
    });

    options.forEach(function (option) {
        option.addEventListener("click", function (e) {
            setSelectTitle(e, selectLabel, dropdown, arrow);
            updateSelectedOption(e.target, selectedOption);
            selectedOption = e.target;
        });
    });
});

function toggleDropdown(dropdown, arrow) {
    dropdown.classList.toggle("hidden");

    if (!dropdown.classList.contains("hidden")) {
        arrow.classList.add("down"); 
    } else {
        arrow.classList.remove("down");
    }
}

function setSelectTitle(e, selectLabel, dropdown, arrow) {
    const labelElement = document.querySelector(`label[for="${e.target.id}"]`).innerText;
    selectLabel.innerText = labelElement;
    toggleDropdown(dropdown, arrow);
}

function updateSelectedOption(option, previousSelectedOption) {
    if (previousSelectedOption) {
        const previousLabel = document.querySelector(`label[for="${previousSelectedOption.id}"]`);
        previousLabel.classList.remove("selected");
    }

    const currentLabel = document.querySelector(`label[for="${option.id}"]`);
    currentLabel.classList.add("selected");
}


function closeFlashMessage(element) {
    const flashMessage = element.closest('.flash-messages');
    flashMessage.style.display = 'none'; // Hide the entire flash message container
  }

  // Auto fade out after 7 seconds
  setTimeout(function() {
    var flashMessages = document.querySelector('.flash-messages');
    if (flashMessages) {
      flashMessages.style.opacity = '0'; // Start fade out
      setTimeout(function() {
        flashMessages.style.display = 'none'; // Hide completely after fading out
      }, 500); // Wait for the fade-out transition to finish
    }
  }, 7000); // 7 seconds before fade starts

  document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting normally

    const formData = new FormData(this);

    // Use Fetch API to submit the form data via AJAX
    fetch('/analyze_generate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.image_path) {
            const imgElement = document.getElementById('generated-image');
            const noPlanMessage = document.getElementById('no-plan-message');
            imgElement.src = data.image_path;
            imgElement.style.display = 'block';
            noPlanMessage.style.display = 'none';

            // Scale the image to fit within the container
            imgElement.onload = function() {
                const maxWidth = document.querySelector('.main').clientWidth;
                const maxHeight = document.querySelector('.main').clientHeight;
                if (imgElement.naturalWidth > maxWidth || imgElement.naturalHeight > maxHeight) {
                    imgElement.style.width = maxWidth + 'px';
                    imgElement.style.height = 'auto';
                }
            };
        }
    })
    .catch(error => console.error('Error:', error));
});