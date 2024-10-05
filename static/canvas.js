var namespace = joint.shapes;

var graph = new joint.dia.Graph({}, { cellNamespace: namespace });

var paper = new joint.dia.Paper({
    el: document.getElementById('myholder'),
    model: graph,
    gridSize: 10,
    drawGrid: true,
    background: {
        color: 'rgba(0, 255, 0, 0.3)'
    },
    cellViewNamespace: namespace
});

// Variable to store the copied element in memory
var copiedElement = null;

// Function to handle shape selection on click
paper.on('element:pointerdown', function(cellView) {
    selectedElement = cellView.model;
    console.log('Shape selected:', selectedElement);
});

// Function to handle copy (Ctrl+C) and save to memory
function copySelectedShape() {
    if (selectedElement) {
        copiedElement = selectedElement.clone(); // Clone the selected shape
        console.log('Shape copied to memory!');
    }
}

// Function to handle paste (Ctrl+V) from memory
function pasteCopiedShape() {
    if (copiedElement) {
        var pastedElement = copiedElement.clone(); // Clone the copied element
        pastedElement.translate(20, 20); // Offset the pasted shape slightly
        pastedElement.addTo(graph); // Add the pasted shape to the graph
        console.log('Shape pasted from memory!');
    }
}

// Function to add rectangle in the center of the screen
function addRectangleInCenter() {
    var paperSize = paper.getComputedSize();
    var centerX = paperSize.width / 2;
    var centerY = paperSize.height / 2;

    var newRect = new joint.shapes.standard.Rectangle();
    newRect.position(centerX - 50, centerY - 20); // Offset to center the rectangle
    newRect.resize(100, 40);
    newRect.attr({
        body: {
            fill: 'orange'
        },
        label: {
            text: 'Center Rect',
            fill: 'white'
        }
    });
    newRect.addTo(graph);
}

// Function to add circle in the center of the screen
function addCircleInCenter() {
    var paperSize = paper.getComputedSize();
    var centerX = paperSize.width / 2;
    var centerY = paperSize.height / 2;

    var newCircle = new joint.shapes.standard.Circle();
    newCircle.position(centerX - 40, centerY - 40); // Offset to center the circle (radius 40)
    newCircle.resize(80, 80); // Set diameter (80x80)
    newCircle.attr({
        body: {
            fill: 'orange'
        },
        label: {
            text: 'Center Circle',
            fill: 'white'
        }
    });
    newCircle.addTo(graph);
}

// Function to add a line (link) between two points
function addLine() {
    var sourcePoint = { x: 50, y: 50 }; // Define starting point of the line
    var targetPoint = { x: 200, y: 200 }; // Define ending point of the line

    var link = new joint.shapes.standard.Link();
    link.source(sourcePoint);
    link.target(targetPoint);
    link.addTo(graph); // Add the line to the graph
    console.log('Line added between points:', sourcePoint, 'and', targetPoint);
}

// Event listeners for the toolbar buttons
document.getElementById('add-rectangle').addEventListener('click', addRectangleInCenter);
document.getElementById('add-circle').addEventListener('click', addCircleInCenter);
document.getElementById('add-line').addEventListener('click', addLine); // Add event listener for line

// Event listener for keydown (Ctrl+C and Ctrl+V)
document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'c') {
        event.preventDefault(); // Prevent default browser copy
        copySelectedShape(); // Call copy function
    }
    if (event.ctrlKey && event.key === 'v') {
        event.preventDefault(); // Prevent default browser paste
        pasteCopiedShape(); // Call paste function
    }
});
