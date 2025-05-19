
document.getElementById("analyze-form").addEventListener("submit", function(e) {
  e.preventDefault();
  document.getElementById("form-box").classList.add("spin-anim");

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
  .then(response => response.json())
  .then(data => {
      console.log("Réponse de l'API :", data);
      const iframe = document.getElementById('result-container');
      iframe.src = "data:text/html;base64," + data.lime_html_base64;
      document.getElementById("form-box").classList.remove("spin-anim");
  })
  .catch(err => {
    console.error(err);
    document.getElementById("form-box").classList.remove("spin-anim");
  });
});
