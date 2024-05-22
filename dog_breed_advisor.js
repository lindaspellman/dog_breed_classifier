function displayInput() {
    // Get the input element
    const inputElement = document.getElementById('myInput');

    // Get the value of the input element
    const inputValue = inputElement.value;

    // Display the value in the output paragraph
    const outputElement = document.getElementById('output');
    outputElement.textContent = `You entered: ${inputValue}`;
}
