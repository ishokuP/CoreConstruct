var width = window.innerWidth;
var height = window.innerHeight;

var stage = new Konva.Stage({
  container: 'container',
  width: width,
  height: height
});

// Create a layer
var layer = new Konva.Layer();
stage.add(layer);

// Variable to store the copied shape and the currently selected shape
var copiedShape = null;
var selectedShape = null;

// Function to clear the selection outline of all shapes
function clearSelection() {
  layer.getChildren().forEach(function(shape) {
    if (shape instanceof Konva.Line) {
      shape.stroke('black');  // Reset stroke to black for lines
    } else {
      shape.stroke(null);  // Remove stroke for other shapes
    }
    shape.strokeWidth(0);  // Reset stroke width for all shapes
  });
  layer.draw();  // Redraw the layer
}

// Function to select a shape and give it an outline
function selectShape(shape) {
  clearSelection();  
  shape.stroke('blue');  
  shape.strokeWidth(2); 
  selectedShape = shape;  
  layer.draw();  // Redraw the layer with the updated selection
}


// Function to copy shape data to the clipboard
async function copyToClipboard() {
  if (selectedShape) {
    // Serialize the selected shape to JSON
    var shapeData = selectedShape.toJSON();
    try {
      // Write JSON string to the system clipboard
      await navigator.clipboard.writeText(shapeData);
      console.log('Shape copied to clipboard:', shapeData);
    } catch (err) {
      console.error('Failed to copy shape to clipboard:', err);
    }
  }
}

// Function to paste shape from the clipboard
async function pasteFromClipboard() {
  try {
    // Read from the clipboard
    const clipboardData = await navigator.clipboard.readText();
    if (clipboardData) {
      // Parse the JSON from the clipboard
      const shapeData = JSON.parse(clipboardData);
      // Recreate the shape from the clipboard data
      var newShape = Konva.Node.create(shapeData);
      newShape.position({
        x: newShape.x() + 20,  // Offset to avoid overlap
        y: newShape.y() + 20
      });
      layer.add(newShape);
      newShape.on('click', function() {
        selectShape(newShape);  // Add click event for newly pasted shape
      });
      layer.draw();
      console.log('Shape pasted from clipboard:', newShape);
    }
  } catch (err) {
    console.error('Failed to paste shape from clipboard:', err);
  }
}

// Function to add a rectangle in the center of the canvas
function addRectangle() {
  var newRect = new Konva.Rect({
    x: stage.width() / 2 - 50,  // Center horizontally
    y: stage.height() / 2 - 50, // Center vertically
    width: 100,
    height: 100,
    fill: 'blue',
    draggable: true
  });
  layer.add(newRect);
  newRect.on('click', function() {
    selectShape(newRect);  // Add click event to select the rectangle
  });
  layer.draw();
}

// Function to add a circle in the center of the canvas
function addCircle() {
  var newCircle = new Konva.Circle({
    x: stage.width() / 2,  // Center horizontally
    y: stage.height() / 2,  // Center vertically
    radius: 50,
    fill: 'red',
    draggable: true
  });
  layer.add(newCircle);
  newCircle.on('click', function() {
    selectShape(newCircle);  // Add click event to select the circle
  });
  layer.draw();
}

// Function to add a straight line in the center of the canvas
function addLine() {
  var newLine = new Konva.Line({
    points: [
      stage.width() / 2 - 50, stage.height() / 2,  // Start point (center)
      stage.width() / 2 + 50, stage.height() / 2   // End point (center)
    ],
    stroke: 'black',
    draggable: true
  });
  layer.add(newLine);
  newLine.on('click', function() {
    selectShape(newLine);  // Add click event to select the line
  });
  layer.draw();
}

// Function to save the diagram to the backend
function saveDiagram() {
  var diagramData = stage.toJSON();  // Export the current stage (canvas) as JSON

  fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ diagram: diagramData })
  })
  .then(response => response.json())
  .then(data => {
    alert(data.message);  // Notify user that the diagram was saved
  });
}

// Function to load the diagram from the backend
function loadDiagram() {
  fetch('/load')
    .then(response => response.json())
    .then(data => {
      if (data.diagram) {
        stage.destroy();  // Clear the current stage
        stage = Konva.Node.create(data.diagram, 'container');  // Load the saved diagram into the stage
      }
    });
}

// Event listeners for the buttons
document.getElementById('add-rectangle').addEventListener('click', addRectangle);
document.getElementById('add-circle').addEventListener('click', addCircle);
document.getElementById('add-line').addEventListener('click', addLine);
document.getElementById('save').addEventListener('click', saveDiagram);
document.getElementById('load').addEventListener('click', loadDiagram);

// Event listeners for copy (Ctrl+C) and paste (Ctrl+V)
document.addEventListener('keydown', function(event) {
  if (event.ctrlKey && event.key === 'c') {
    event.preventDefault();
    copyToClipboard();  // Copy the selected shape
  }
  if (event.ctrlKey && event.key === 'v') {
    event.preventDefault();
    pasteFromClipboard();  // Paste the shape from clipboard
  }
});


// Assuming we have extracted floor plan data as an array of architectural components
const floorPlanData = [
  { type: 'wall', x: 50, y: 100, width: 200, height: 10 },
  { type: 'door', x: 150, y: 120, width: 40, height: 10 },
  { type: 'window', x: 70, y: 130, width: 60, height: 5 },
  // Add other elements as needed...
];

// Function to convert the extracted floor plan elements into Konva shapes
function renderFloorPlan(data) {
  data.forEach(element => {
    let shape;
    
    switch (element.type) {
      case 'wall':
        // Add wall as a Konva rectangle
        shape = new Konva.Rect({
          x: element.x,
          y: element.y,
          width: element.width,
          height: element.height,
          fill: 'gray',  // Use appropriate color for walls
          draggable: true
        });
        break;

      case 'door':
        // Add door as a Konva line or rectangle
        shape = new Konva.Rect({
          x: element.x,
          y: element.y,
          width: element.width,
          height: element.height,
          fill: 'brown',  // Use appropriate color for doors
          draggable: true
        });
        break;

      case 'window':
        // Add window as a Konva line or rectangle
        shape = new Konva.Rect({
          x: element.x,
          y: element.y,
          width: element.width,
          height: element.height,
          fill: 'lightblue',  // Use appropriate color for windows
          draggable: true
        });
        break;

      // Add more cases for other components like furniture, rooms, etc.
    }

    // Add the shape to the Konva layer
    layer.add(shape);
  });

  // Redraw the layer to reflect the new shapes
  layer.draw();
}

// Call the render function with the extracted data
renderFloorPlan(floorPlanData);
