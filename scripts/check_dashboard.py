"""Check dashboard integrity."""
import re

js = open('dashboard/app.js', 'r', encoding='utf-8').read()
html = open('dashboard/index.html', 'r', encoding='utf-8').read()

# Check all element IDs referenced in JS exist in HTML
js_ids = set(re.findall(r"getElementById\(['\"](\w+)['\"]\)", js))
html_ids = set(re.findall(r'id="(\w+)"', html))
missing = js_ids - html_ids
if missing:
    print(f"JS references IDs not in HTML: {missing}")
else:
    print(f"All {len(js_ids)} JS element references found in HTML")

# Check fetch endpoints in JS
endpoints = re.findall(r"fetchJSON\(['\"]([^'\"]+)['\"]\)", js)
print(f"\nFetch endpoints in JS ({len(endpoints)}):")
for ep in sorted(set(endpoints)):
    print(f"  {ep}")

# Check for broken API_BASE usage
api_base_refs = len(re.findall(r'API_BASE', js))
print(f"\nAPI_BASE references: {api_base_refs}")

# Check CSS for unclosed braces
css = open('dashboard/styles.css', 'r', encoding='utf-8').read()
opens = css.count('{')
closes = css.count('}')
print(f"\nCSS braces: {{ {opens} }} {closes} {'OK' if opens == closes else 'MISMATCH!'}")
