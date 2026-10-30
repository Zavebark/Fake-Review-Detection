(() => {
  const API = "http://127.0.0.1:8000";
  const BADGE_CLASS = "rrc-badge";
  const DONE = "data-rrc-processed";
  const SELECTORS = [
    '[data-review-id]', '[data-testid*="review"]', '[class*="review"]',
    '[itemprop="review"]', 'article'
  ];
  const seen = new WeakSet();
  let enabled = true;
  let timer = null;

  chrome.storage.sync.get({enabled:true}, data => { enabled=data.enabled; if(enabled) scan(); });
  chrome.storage.onChanged.addListener((changes, area) => {
    if(area==="sync" && changes.enabled){enabled=changes.enabled.newValue;if(enabled)scan();else removeBadges();}
  });

  function likelyReview(el) {
    if(!el || el.closest(".rrc-badge")) return false;
    const text=(el.innerText||"").trim();
    if(text.length<35 || text.length>5000) return false;
    // Avoid treating a whole page/container as a review.
    if(el.children.length>12) return false;
    return true;
  }
  function candidates() {
    const found=new Set();
    for(const selector of SELECTORS) {
      document.querySelectorAll(selector).forEach(el=>{
        if(likelyReview(el)) {
          // Prefer the smallest matching review-like element.
          const child=[...el.children].find(c=>likelyReview(c));
          found.add(child||el);
        }
      });
    }
    return [...found].filter(el=>!([...found].some(other=>other!==el && other.contains(el) && (other.innerText||"").length>(el.innerText||"").length*1.4)));
  }
  function extractRating(el) {
    const ratingEl=el.querySelector('[itemprop="ratingValue"],[aria-label*="out of 5"],[aria-label*="stars"]');
    if(!ratingEl)return undefined;
    const raw=ratingEl.getAttribute("content")||ratingEl.getAttribute("aria-label")||ratingEl.textContent;
    const m=String(raw).match(/\d+(?:\.\d+)?/);return m?Number(m[0]):undefined;
  }
  function addBadge(el, result) {
    let badge=el.querySelector(":scope > ."+BADGE_CLASS);
    if(!badge){badge=document.createElement("span");badge.className=BADGE_CLASS;el.appendChild(badge);}
    badge.textContent=`${Number(result.score).toFixed(1)} • ${result.label}`;
    badge.dataset.score=result.score;
    badge.title=(result.explanation_cues||[]).join(", ")+" — estimate only";
    badge.classList.toggle("rrc-human",result.score>=7);
    badge.classList.toggle("rrc-fake",result.score<5);
    el.classList.add("rrc-review-host");
  }
  async function score(el) {
    if(seen.has(el) || el.hasAttribute(DONE)) return;
    const text=(el.innerText||"").trim();
    if(text.length<35)return;
    seen.add(el);el.setAttribute(DONE,"1");
    try {
      const response=await fetch(`${API}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,rating:extractRating(el)})});
      if(!response.ok) throw new Error("API unavailable");
      addBadge(el,await response.json());
    } catch(e) {
      el.removeAttribute(DONE); seen.delete(el);
      // API may not be running yet; avoid flooding the console.
    }
  }
  function scan(){if(!enabled)return;candidates().forEach(score);}
  function removeBadges(){document.querySelectorAll("."+BADGE_CLASS).forEach(b=>b.remove());document.querySelectorAll(".rrc-review-host").forEach(e=>e.classList.remove("rrc-review-host"));}
  const observer=new MutationObserver(()=>{if(!enabled)return;clearTimeout(timer);timer=setTimeout(scan,450);});
  observer.observe(document.documentElement,{childList:true,subtree:true});
  scan();
})();