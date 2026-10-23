"""Quasi-identifier extraction + hierarchies shared by IA-2 Modules 4 and 5."""
import json, re
ROOT = "."
def clean_part(p):
    p = re.sub(r"\{\{.*", "", p); p = re.sub(r"\[https?:[^\]]*\]?", "", p); p = re.sub(r"\(.*?\)|\)", "", p)
    return p.strip()
CONTINENT = {}
for c in ["Norway","Finland","Sweden","Denmark","Netherlands","Spain","France","Italy","Russia","Romania","Estonia","West Germany","Faroe Islands","Soviet Union","Uzbek SSR","United Kingdom"]: CONTINENT[c] = "Europe"
for c in ["India","Nepal","Myanmar","Taiwan","Saudi Arabia","Philippines","Vietnam","China","Afghanistan","Bahrain","Pakistan","Republic of China","Malaysia","Iran","Federation of Malaya","British India"]: CONTINENT[c] = "Asia"
for c in ["Morocco","Nigeria","South Africa","Egypt","Algeria","Liberia","Cameroon","Tanzania","Togo","Chad","Ghana","Kenya"]: CONTINENT[c] = "Africa"
for c in ["United States","Canada","Argentina","Brazil","Jamaica","Mexico","Ecuador","Chile","Dominican Republic","Netherlands Antilles","American Samoa","Guam"]: CONTINENT[c] = "Americas"
for c in ["Australia","New Zealand","Papua New Guinea"]: CONTINENT[c] = "Oceania"
US_STATES = {"California","Louisiana","Florida","Texas","New York City","New York","Georgia","Colorado","Oklahoma","Kentucky"}
CITY_COUNTRY = {"Bergen":"Norway","Bærum":"Norway","Helsinki":"Finland","Lahti":"Finland","Tartu":"Estonia","Emmen":"Netherlands","North Brabant":"Netherlands","Delhi":"India","Chittorgarh":"India"}
COUNTRY = {"U.S.":"United States","US":"United States","USA":"United States","United States":"United States","U.S":"United States",
           "England":"United Kingdom","Scotland":"United Kingdom","Wales":"United Kingdom","Northern Ireland":"United Kingdom","UK":"United Kingdom","U.K.":"United Kingdom"}
def country_of(parts):
    c = parts[-1]
    if c in US_STATES: return "United States"
    if c in CITY_COUNTRY: return CITY_COUNTRY[c]
    return COUNTRY.get(c, c)

def load_qi_records():
    raw = [json.loads(l) for l in open(f"{ROOT}/data/raw_blp_data.jsonl", encoding="utf-8")]
    recs = []
    for r in raw:
        bd, bp = r["infobox"].get("birth_date",""), r["infobox"].get("birth_place","")
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", bd or "")
        parts = [q for q in (clean_part(p) for p in re.sub(r"\[\[|\]\]","",bp).split(",")) if q]
        if m and parts: recs.append({"wiki_id":r["wiki_id"],"title":r["title"],"y":int(m.group(1)),"mo":int(m.group(2)),"iso":bd,"parts":parts})
    return recs
