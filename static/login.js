const loginForm = document.querySelector("#login-form");
const registerForm = document.querySelector("#register-form");
const errorEl = document.querySelector("#auth-error");
const showLogin = document.querySelector("#show-login");
const showRegister = document.querySelector("#show-register");

function showError(message) {
  errorEl.hidden = !message;
  errorEl.textContent = message || "";
}

function selectMode(register) {
  loginForm.hidden = register;
  registerForm.hidden = !register;
  showLogin.classList.toggle("primary", !register);
  showRegister.classList.toggle("primary", register);
  showError("");
}

showLogin.addEventListener("click", () => selectMode(false));
showRegister.addEventListener("click", () => selectMode(true));

async function submit(url, payload) {
  showError("");
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    showError(data.error || "Impossible de continuer.");
    return;
  }
  window.location.href = "/";
}

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(loginForm);
  submit("/api/login", {
    email: form.get("email"),
    password: form.get("password"),
  });
});

registerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(registerForm);
  submit("/api/register", {
    name: form.get("name"),
    email: form.get("email"),
    password: form.get("password"),
  });
});
