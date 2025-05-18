
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
  .then(data => {
      console.log(data)
      const container = document.getElementById('result-container');
      container.innerHTML = `
          <div class="progress-bar">
              <div class="result-bar">
                   <div class="result-bar-fill" style="width: ${data.percent}%;">${data.label} ${data.percent}%</div>
              </div>
          </div>
          <div class="result-explanation">
              Possession her thoroughly remarkably terminated man continuing. Removed greater to do ability. You shy shall while but wrote marry. Call why sake has sing pure. Gay six set polite nature worthy. So matter be me we wisdom should basket moment merely. Me burst ample wrong which would mr he could. Visit arise my point timed drawn no. Can friendly laughter goodness man him appetite carriage. Any widen see gay forth alone fruit bed.
          </div>
       `;
      container.scrollIntoView({ behavior: "smooth" });
  })
  .catch(err => {
    console.error("Erreur API :", err);
  });
});