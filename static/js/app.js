(() => {
  const form = document.getElementById("uploadForm");
  const input = document.getElementById("imageInput");
  const dropzone = document.getElementById("dropzone");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const loading = document.getElementById("loading");
  const btn = document.getElementById("btnSubmit");

  if (!form || !input || !dropzone) return;

  const showPreview = (file) => {
    if (!file) return;
    previewWrap.style.display = "flex";
    previewName.textContent = `${file.name} • ${(file.size/1024/1024).toFixed(2)} MB`;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
  };

  input.addEventListener("change", (e) => {
    showPreview(e.target.files?.[0]);
  });

  // Drag & drop
  ["dragenter","dragover"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave","drop"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    input.files = e.dataTransfer.files;
    showPreview(file);
  });

  // Loading state
  form.addEventListener("submit", () => {
    loading.style.display = "flex";
    btn.disabled = true;
    btn.textContent = "Analyzing…";
  });
})();
