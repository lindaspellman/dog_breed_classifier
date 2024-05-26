function displayInput() {
    // Get the input element
    const inputElement = document.getElementById('myInput');

    // Get the value of the input element
    const inputValue = inputElement.value;

    // Display the value in the output paragraph
    const outputElement = document.getElementById('example-output');
    outputElement.textContent = `You entered: ${inputValue}`;
}

document.addEventListener('DOMContentLoaded', function() {
    const submitButton = document.getElementById('submitBtn');
    const resultsDiv = document.getElementById('results');

    submitButton.addEventListener('click', async function(event) {
        event.preventDefault(); // Prevent form from submitting normally
        
        const number1 = document.getElementById('number1').value;
        const number2 = document.getElementById('number2').value;
        const number3 = document.getElementById('number3').value;
        const number4 = document.getElementById('number4').value;

        
        const dataToSend = {
            number1: number1,
            number2: number2,
            number3: number3,
            number4: number4
        };

        try {
            const response = await fetch('https://yourserver.com/api/endpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataToSend)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            resultsDiv.innerHTML = 'Error fetching results. Please try again later.';
        }
    });

    function displayResults(data) {
        // Process and display the machine learning results
        resultsDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
});


// async function predict() {
//     const features = [
//         parseFloat(document.getElementById('feature1').value),
//         parseFloat(document.getElementById('feature2').value),
//         parseFloat(document.getElementById('feature3').value),
//         parseFloat(document.getElementById('feature4').value)
//     ];

//     try {
//         const response = await fetch('http://localhost:5000/predict', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({ features: features })
//         });
//         const data = await response.json();
//         document.getElementById('prediction-output').textContent = `Prediction: ${data.prediction}`;
//     } catch (error) {
//         console.error('Error fetching prediction:', error);
//         document.getElementById('prediction-output').textContent = 'Failed to fetch prediction.';
//     }
// }
