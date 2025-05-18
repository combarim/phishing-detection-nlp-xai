
document.getElementById("analyze-form").addEventListener("submit", function(e) {
  e.preventDefault();

  const formData = new FormData(this);

  const data = {
    email_object: formData.get("email_object"),
    email_text: formData.get("email_text")
  };

  fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
  .then(res => res.json())
  .then(fullHtml => {
      const container = document.getElementById('result-container');
      const parser = new DOMParser();
      const doc = parser.parseFromString(fullHtml, 'text/html');
      container.innerHTML = doc.body.innerHTML;
      container.scrollIntoView({ behavior: "smooth" });

  })
  .catch(err => {
    console.error("Erreur API :", err);
  });
});