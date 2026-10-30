const toggle=document.querySelector("#toggle");const status=document.querySelector("#status");
chrome.storage.sync.get({enabled:true},({enabled})=>{toggle.checked=enabled;render(enabled)});
function render(on){status.textContent=on?"On — review scores are enabled.":"Off — review scores are hidden."}
toggle.addEventListener("change",()=>{const enabled=toggle.checked;chrome.storage.sync.set({enabled});render(enabled);});