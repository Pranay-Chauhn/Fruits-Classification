document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("imageFile");
    const file = fileInput.files[0];

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Network response was not ok " + response.statusText);
        }

        const result = await response.json();

        // Set the predicted image source
        const predictedImage = document.getElementById("predictedImage");
        const reader = new FileReader();

        // Read the uploaded file as a data URL
        reader.onload = function (e) {
            predictedImage.src = e.target.result; // Set the src to the data URL
            predictedImage.style.display = 'block'; // Make the image visible
        }
        reader.readAsDataURL(file); // Convert the file to a data URL

        document.getElementById("predictedClass").textContent = `Class: ${result.class}`;
        document.getElementById("confidence").textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("predictedClass").textContent = "Error: Unable to predict.";
    }
});

