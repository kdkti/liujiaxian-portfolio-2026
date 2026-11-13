const views = [...document.querySelectorAll("[data-view]")];
const viewById = new Map(views.map(view => [view.id, view]));
const tabs = [...document.querySelectorAll("[data-tab]")];
const themeButton = document.querySelector(".theme-toggle");
const root = document.documentElement;
let activeView;

function showView(moveFocus = false) {
  const id = location.hash.slice(1);
  const view = viewById.get(id) ?? viewById.get("cases");
  if (view === activeView) return;

  views.forEach(section => { section.hidden = section !== view; });
  const tabId = id.startsWith("case-") || id === "daily-activity" ? "cases" : view.id;
  tabs.forEach(tab => {
    if (tab.dataset.tab === tabId) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  });
  activeView = view;

  if (moveFocus) {
    view.querySelector("[tabindex='-1']")?.focus({preventScroll: true});
    if (matchMedia("(max-width: 760px)").matches) {
      document.querySelector(".toolbar").scrollIntoView({block: "start"});
    } else {
      window.scrollTo(0, 0);
    }
  }
}

function setTheme(theme) {
  const light = theme === "light";
  root.dataset.theme = light ? "light" : "dark";
  themeButton.setAttribute("aria-label", light ? "切换为深色模式" : "切换为浅色模式");
  themeButton.setAttribute("aria-pressed", String(light));
  themeButton.querySelector("[data-theme-icon]").textContent = light ? "☾" : "☀";
  themeButton.querySelector("[data-theme-label]").textContent = light ? "深色" : "浅色";
  dispatchEvent(new Event("portfolio-theme"));
}

let savedTheme;
try { savedTheme = localStorage.getItem("liujiaxian-vcard-theme"); } catch {}
setTheme(savedTheme);
themeButton.hidden = false;
themeButton.addEventListener("click", () => {
  const theme = root.dataset.theme === "light" ? "dark" : "light";
  setTheme(theme);
  try { localStorage.setItem("liujiaxian-vcard-theme", theme); } catch {}
});

showView();
addEventListener("hashchange", () => showView(true));
