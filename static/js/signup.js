document.addEventListener("DOMContentLoaded", function () {

const form = document.getElementById("signupForm");

const password = document.getElementById("password");

const confirmPassword =
    document.getElementById("confirm_password");

const passwordMessage =
    document.getElementById("passwordMessage");

const confirmMessage =
    document.getElementById("confirmMessage");


// Check password while typing
password.addEventListener("input", function () {

    if (password.value.length === 0) {

        passwordMessage.textContent =
            "Password must contain at least 6 characters.";

    } else if (password.value.length < 6) {

        passwordMessage.textContent =
            "Password is too short.";

    } else {

        passwordMessage.textContent =
            "Password length is valid.";

    }

});


// Check whether passwords match
confirmPassword.addEventListener("input", function () {

    if (confirmPassword.value === "") {

        confirmMessage.textContent = "";

    } else if (password.value === confirmPassword.value) {

        confirmMessage.textContent =
            "Passwords match.";

    } else {

        confirmMessage.textContent =
            "Passwords do not match.";

    }

});


// Final validation before submitting
form.addEventListener("submit", function (event) {

    if (password.value.length < 6) {

        event.preventDefault();

        alert("Password must contain at least 6 characters.");

        password.focus();

        return;
    }


    if (password.value !== confirmPassword.value) {

        event.preventDefault();

        alert("Passwords do not match.");

        confirmPassword.focus();

        return;
    }

});


});

// Show / Hide Password
function togglePassword(inputId, button) {
const input = document.getElementById(inputId);

if (input.type === "password") {

    input.type = "text";

    button.textContent = "Hide";

} else {

    input.type = "password";

    button.textContent = "Show";
}

}
