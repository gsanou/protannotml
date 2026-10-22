/* ProtAnnotML Academy — shared runtime helpers.
   Loaded synchronously in <head> so that the MathJax config exists before MathJax loads. */

window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], tags: "none" },
  svg: { fontCache: "global" },
  options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] }
};

const PA = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  /** Create an SVG element with attributes. */
  function svg(tag, attrs = {}, parent = null) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") el.textContent = v;
      else el.setAttribute(k, v);
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  /** Create an HTML element with attributes. */
  function html(tag, attrs = {}, parent = null) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v);
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  /** Read a CSS custom property from :root (theme-aware at call time). */
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  const reducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /** Build the table of contents from h2[id] inside .article. */
  function buildToc() {
    const toc = document.querySelector(".toc ol");
    if (!toc) return;
    const heads = [...document.querySelectorAll(".article h2[id]")];
    const links = heads.map((h) => {
      const li = html("li", {}, toc);
      const a = html("a", { href: "#" + h.id }, li);
      a.textContent = h.dataset.short || h.textContent.replace(/^\s*\d+\s*/, "");
      return a;
    });
    if (!("IntersectionObserver" in window)) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === "#" + e.target.id));
          }
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    heads.forEach((h) => obs.observe(h));
  }

  /** Wrap <pre data-file> blocks with a header and a copy button; highlight with hljs. */
  function decorateCode() {
    document.querySelectorAll("pre[data-lang]").forEach((pre) => {
      const wrap = html("div", { class: "codeblock" });
      pre.parentNode.insertBefore(wrap, pre);
      const head = html("div", { class: "codehead" }, wrap);
      html("span", { text: pre.dataset.file || pre.dataset.lang }, head);
      const btn = html("button", { class: "copy-btn", type: "button", text: "Copier" }, head);
      wrap.appendChild(pre);
      const code = pre.querySelector("code") || pre;
      btn.addEventListener("click", () => {
        const text = code.innerText;
        const done = () => { btn.textContent = "Copié"; setTimeout(() => (btn.textContent = "Copier"), 1400); };
        try {
          navigator.clipboard.writeText(text).then(done, () => selectText(code));
        } catch (_) { selectText(code); }
      });
      if (window.hljs && pre.dataset.lang !== "text") {
        try { window.hljs.highlightElement(code); } catch (_) { /* keep plain text */ }
      }
    });
  }

  function selectText(node) {
    const range = document.createRange();
    range.selectNodeContents(node);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  /** Simple step player for animations: calls render(step) on each tick. */
  function stepper({ nSteps, render, playBtn, prevBtn, nextBtn, resetBtn, interval = 1400, label }) {
    let step = 0;
    let timer = null;
    const show = () => {
      render(step);
      if (label) label.textContent = `${step + 1} / ${nSteps}`;
      if (prevBtn) prevBtn.disabled = step === 0;
      if (nextBtn) nextBtn.disabled = step === nSteps - 1;
    };
    const stop = () => { clearInterval(timer); timer = null; if (playBtn) playBtn.textContent = "Lecture"; };
    const play = () => {
      if (timer) return stop();
      if (step === nSteps - 1) step = 0;
      if (playBtn) playBtn.textContent = "Pause";
      show();
      timer = setInterval(() => {
        if (step >= nSteps - 1) return stop();
        step += 1; show();
      }, reducedMotion() ? interval * 1.5 : interval);
    };
    playBtn && playBtn.addEventListener("click", play);
    prevBtn && prevBtn.addEventListener("click", () => { stop(); step = Math.max(0, step - 1); show(); });
    nextBtn && nextBtn.addEventListener("click", () => { stop(); step = Math.min(nSteps - 1, step + 1); show(); });
    resetBtn && resetBtn.addEventListener("click", () => { stop(); step = 0; show(); });
    show();
    return { show, goto: (s) => { step = s; show(); } };
  }

  /** Re-run a renderer when the color scheme changes so SVG colors follow the theme. */
  function onTheme(fn) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener && mq.addEventListener("change", fn);
    new MutationObserver(fn).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildToc();
    decorateCode();
  });

  return { svg, html, token, reducedMotion, stepper, onTheme };
})();
