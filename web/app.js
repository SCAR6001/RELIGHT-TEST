async function generate() {
  const file = document.getElementById("image").files[0];
  if (!file) return alert("Upload an image");

  const fd = new FormData();
  fd.append("image", file);
  fd.append("direction", document.getElementById("direction").value);
  fd.append("brightness", document.getElementById("brightness").value);

  const res = await fetch("/api/relight", { method: "POST", body: fd });
  const data = await res.json();

  document.getElementById("result").innerHTML =
    `<img src="${data.url}" width="100%">`;
}
