// The profile markup is in index.html; keep its styles separate from the game.
import "./home.css";

function setupContactForm() {
  const form = document.querySelector<HTMLFormElement>("#contact-form");
  const status = document.querySelector<HTMLParagraphElement>("#contact-status");
  const button = form?.querySelector<HTMLButtonElement>('button[type="submit"]');

  if (!form || !status || !button) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    // Submit in the background instead of navigating to Formspree.
    event.preventDefault();
    if (button.disabled) {
      return;
    }

    button.disabled = true;
    status.textContent = "Sending...";
    status.classList.remove("text-success", "text-danger");

    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      });

      // fetch only rejects network errors, so check HTTP errors separately.
      if (!response.ok) {
        throw new Error(`Submission failed: HTTP ${response.status}`);
      }

      status.textContent = "Thank you! Your message has been submitted.";
      status.classList.add("text-success");
      form.reset();
    } catch (error) {
      // Keep the entered message so the visitor can retry.
      status.textContent =
        "We couldn't confirm your submission. Please try again shortly.";
      status.classList.add("text-danger");
      console.error("Contact form submission failed:", error);
    } finally {
      button.disabled = false;
    }
  });
}

setupContactForm();
