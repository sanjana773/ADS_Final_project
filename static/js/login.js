const loginForm = document.getElementById("loginForm");
const message = document.getElementById("message");

loginForm.addEventListener("submit", function(event) {

    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();

    // Check empty fields
    if (username === "" || password === "") {

        event.preventDefault();

        message.textContent = "Please enter email and password.";
        message.style.color = "red";

        return;
    }


    // Check Gmail format
    const gmailPattern = /^[a-zA-Z0-9._%+-]+@gmail\.com$/;

    if (!gmailPattern.test(username)) {

        event.preventDefault();

        message.textContent =
            "Please enter a valid Gmail address, example@gmail.com";

        message.style.color = "red";

        return;
    }


    // Check password length
    if (password.length < 6) {

        event.preventDefault();

        message.textContent =
            "Password must be at least 6 characters.";

        message.style.color = "red";

        return;
    }


    /*
       Everything is valid.

       Do NOT use event.preventDefault() here.

       The form will now be submitted to Flask,
       and Flask will check whether the account exists.
    */

});
