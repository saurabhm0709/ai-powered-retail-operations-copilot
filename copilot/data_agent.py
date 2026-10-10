"""[3B] Data Agent: pandas tools over the CSVs + deterministic dispatcher (ARCHITECTURE.md §6 [3B])."""

import difflib
import re
from functools import lru_cache

import pandas as pd

from copilot import config
from copilot.data_store import before_anchor, load_table, within_window
from copilot.schemas import DataResult, Facts, TriageResult

MAX_SKU_CANDIDATES = 3
# Words that do not identify a product: the store sells "sports goods", "Size 5" is a size, "set"/"pair" a quantity.
SKU_STOPWORDS = {"the", "and", "for", "with", "sports", "goods", "item", "product", "size", "set", "pair"}
TYPO_CUTOFF = 0.8  # difflib similarity needed to correct a misspelt word ("tenis" -> "tennis")


# Lowercase words of 3+ letters minus stopwords (digits dropped, so "Size 5" does not need to be typed).
def name_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower())) - SKU_STOPWORDS


# All words used in product names (the vocabulary for mention finding and typo correction).
@lru_cache(maxsize=1)
def product_vocabulary() -> frozenset[str]:
    vocabulary = set()
    for name in load_table("product_master")["name"]:
        vocabulary |= name_words(name)
    return frozenset(vocabulary)


# Free text -> longest run of consecutive product-name words ("Tennis Ball Pack"), or None.
# Used by the keyword (mock) triage; the LLM triage extracts product_mention itself.
def find_product_mention(text: str) -> str | None:
    vocabulary = product_vocabulary()
    best_run, current_run = [], []
    for word in re.findall(r"[A-Za-z]+", text):
        if word.lower() in vocabulary:
            current_run.append(word)
            if len(current_run) > len(best_run):
                best_run = list(current_run)
        else:
            current_run = []
    return " ".join(best_run) if best_run else None


# Products whose name contains ALL the given words.
def products_with_all_words(words: set[str]) -> pd.DataFrame:
    products = load_table("product_master")
    contains_all = products["name"].apply(lambda name: words <= name_words(name))
    return products[contains_all]


# Mention words -> same words with typos fixed to the closest product-name word (difflib, stdlib).
def correct_typos(words: set[str]) -> set[str]:
    corrected = set()
    for word in words:
        close = difflib.get_close_matches(word, product_vocabulary(), n=1, cutoff=TYPO_CUTOFF)
        corrected.add(close[0] if close else word)
    return corrected


# Matching products -> the one core SKU (not an SKU8xxx variant) whose whole name the mention covers, else None.
# "Match Football" covers core "Match Football Size 5"; "running" does not cover "Running Shoes" -> ambiguous.
def covered_core_sku(matches: pd.DataFrame, words: set[str]) -> str | None:
    core = matches[~matches["sku"].str.startswith("SKU8")]
    covered = core[core["name"].apply(lambda name: name_words(name) <= words)]
    return covered["sku"].iloc[0] if len(covered) == 1 else None


# Product mention -> (sku or None, candidate skus). ARCHITECTURE.md §6 [2] step 3: exact name, else all
# mention words in the name (typo fallback only if nothing matches), prefer core, else up to 3 candidates.
def resolve_sku(mention: str) -> tuple[str | None, list[str]]:
    products = load_table("product_master")
    exact = products[products["name"].str.lower() == mention.strip().lower()]
    if len(exact) == 1:
        return exact["sku"].iloc[0], []
    words = name_words(mention)
    if not words:
        return None, []
    matches = products_with_all_words(words)
    if len(matches) == 0:
        words = correct_typos(words)
        matches = products_with_all_words(words)
    if len(matches) == 1:
        return matches["sku"].iloc[0], []
    core_sku = covered_core_sku(matches, words)
    if core_sku:
        return core_sku, []
    # Core SKUs first so the follow-up question offers the main product before its variants.
    ranked = matches.assign(is_variant=matches["sku"].str.startswith("SKU8")).sort_values(["is_variant", "sku"])
    return None, ranked["sku"].head(MAX_SKU_CANDIDATES).tolist()


# --- Shared lookups ---------------------------------------------------------------------------------

INVENTORY_CSV = "inventory.csv"
PRODUCT_CSV = "product_master.csv"
RECEIPTS_CSV = "receipts.csv"
VENDOR_CSV = "vendor_master.csv"
CASES_CSV = "support_cases.csv"
RETURNS_CSV = "returns_quality.csv"
STORES_CSV = "store_master.csv"
MATRIX_CSV = "escalation_matrix.csv"

OPEN_STATUSES = ["Open", "In Progress"]
# 04_vendor_quality_policy.md#product-quality-complaints (ARCHITECTURE.md §6 [3B] keyword list)
QUALITY_PATTERN = r"seam|split|crack|warp|torn|defect|quality|lost air|slipp|loose|malfunction|unsafe|injur"
# 04_vendor_quality_policy.md#product-quality-complaints: injury, recall, unsafe or demo-equipment malfunction is P1
SAFETY_ROW_PATTERN = r"injur|unsafe|recall|hazard|malfunction|during demo|safety escalation"


# SKU -> product_master row as a plain dict (name, vendor_id, list_price, high_demand, ...).
def product_row(sku: str) -> dict:
    products = load_table("product_master")
    return products[products["sku"] == sku].iloc[0].to_dict()


# SKU -> product name from product_master.
def product_name(sku: str) -> str:
    return product_row(sku)["name"]


# SKU -> True if product_master marks it high-demand (rules.is_high_demand reads facts["high_demand"]).
def is_high_demand(sku: str) -> bool:
    return bool(product_row(sku)["high_demand"])


# Store + SKU -> inventory row as a dict, or None when the store has no record for the SKU.
def inventory_row(store_id: str, sku: str) -> dict | None:
    inventory = load_table("inventory")
    rows = inventory[(inventory["store_id"] == store_id) & (inventory["sku"] == sku)]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


# Table column -> plain {label: int} counts (JSON-safe for the evidence pack), largest first.
def count_by(table: pd.DataFrame, column: str) -> dict:
    counts = {}
    for label, count in table[column].value_counts().items():
        counts[str(label)] = int(count)
    return counts


# Lists -> one list with duplicates removed, first-seen order kept.
def merge_unique(lists: list[list]) -> list:
    merged = []
    for items in lists:
        for item in items:
            if item not in merged:
                merged.append(item)
    return merged


# --- Inventory tools: 03_inventory_replenishment_policy.md ------------------------------------------

# Inventory numbers -> stock finding codes (#stock-outs-and-replenishment, #inventory-discrepancy).
def stock_findings(on_hand: int, system_qty: int, reorder_point: int, high_demand: bool) -> list[str]:
    findings = []
    if on_hand == 0:
        findings.append("stockout")
        if high_demand:
            findings.append("stockout_high_demand")
    if on_hand <= reorder_point:
        findings.append("below_reorder")
    gap = abs(system_qty - on_hand)
    if gap > config.DISCREPANCY_UNITS or gap > config.DISCREPANCY_PCT * max(system_qty, 1):
        findings.append("inventory_discrepancy")
        # The system offers stock the shelf does not have: customers see false availability (#inventory-discrepancy).
        if on_hand == 0 and system_qty > 0:
            findings.append("false_availability")
    return findings


# Store + SKU -> on-hand, system qty, reorder point, high-demand flag and stock findings.
def stock_status(store_id: str, sku: str) -> DataResult:
    product = product_row(sku)
    high_demand = bool(product["high_demand"])
    facts = {"store_id": store_id, "sku": sku, "product": product["name"], "high_demand": high_demand}
    row = inventory_row(store_id, sku)
    if row is None:
        return DataResult(tool="stock_status", facts={**facts, "has_record": False}, findings=[],
                          summary_lines=[f"No inventory record for {sku} at {store_id}."],
                          data_used=[INVENTORY_CSV, PRODUCT_CSV])
    on_hand, system_qty, reorder_point = int(row["on_hand"]), int(row["system_qty"]), int(row["reorder_point"])
    facts.update(has_record=True, on_hand=on_hand, system_qty=system_qty, reorder_point=reorder_point,
                 last_count_date=str(row["last_count_date"].date()))
    line = (f"{store_id} {product['name']} ({sku}): on_hand {on_hand}, system_qty {system_qty}, "
            f"reorder point {reorder_point}, high-demand: {'yes' if high_demand else 'no'}.")
    return DataResult(tool="stock_status", facts=facts, summary_lines=[line],
                      findings=stock_findings(on_hand, system_qty, reorder_point, high_demand),
                      data_used=[INVENTORY_CSV, PRODUCT_CSV])


# One receipts row -> small JSON-safe dict with its fill percentage (received / ordered).
def receipt_summary(row: dict) -> dict:
    fill_pct = round(row["received_qty"] / row["po_qty"], 2) if row["po_qty"] else 0.0
    return {"receipt_id": row["receipt_id"], "date": str(row["receipt_date"].date()),
            "po_qty": int(row["po_qty"]), "received_qty": int(row["received_qty"]), "fill_pct": fill_pct,
            "quality_check_ok": bool(row["quality_check_ok"]), "damage_notes": row["damage_notes"]}


# Receipt summaries, newest first -> finding codes (04_vendor_quality_policy.md#short-shipments).
def receipt_findings(receipts: list[dict]) -> list[str]:
    is_short = [receipt["fill_pct"] < config.SHORT_SHIPMENT_PCT for receipt in receipts]
    findings = []
    if any(is_short):
        findings.append("short_shipment")
    if len(is_short) >= 2 and is_short[0] and is_short[1]:
        findings.append("consecutive_short_shipment")
    for receipt in receipts:
        # A failed check that only mentions packaging is not a product quality issue.
        if not receipt["quality_check_ok"] and "packaging" not in receipt["damage_notes"].lower():
            findings.append("receipt_quality_issue")
            break
    return findings


# Store + SKU -> last n receipts dated on or before the anchor, with fill % and short-shipment findings.
def receipt_history(store_id: str, sku: str, n: int = config.RECEIPT_HISTORY_N) -> DataResult:
    receipts = before_anchor(load_table("receipts"), "receipt_date")
    rows = receipts[(receipts["store_id"] == store_id) & (receipts["sku"] == sku)]
    rows = rows.sort_values("receipt_date", ascending=False).head(n)
    summaries = [receipt_summary(row) for row in rows.to_dict("records")]
    lines = []
    for receipt in summaries:
        lines.append(f"{receipt['receipt_id']} ({receipt['date']}): received {receipt['received_qty']} of "
                     f"{receipt['po_qty']} ordered ({receipt['fill_pct']:.0%}).")
    if not summaries:
        lines.append(f"No receipts for {sku} at {store_id} up to the anchor time.")
    facts = {"store_id": store_id, "sku": sku, "high_demand": is_high_demand(sku), "receipts": summaries}
    return DataResult(tool="receipt_history", facts=facts, findings=receipt_findings(summaries),
                      summary_lines=lines, data_used=[RECEIPTS_CSV], evidence_rows=summaries)


# SKU -> warehouse on-hand and latest warehouse receipt; depleted if either shows zero.
def warehouse_status(sku: str) -> DataResult:
    row = inventory_row(config.WAREHOUSE_ID, sku)
    latest_receipts = receipt_history(config.WAREHOUSE_ID, sku, n=1).facts["receipts"]
    latest = latest_receipts[0] if latest_receipts else None
    on_hand = int(row["on_hand"]) if row else None
    depleted = on_hand == 0 or (latest is not None and latest["received_qty"] == 0)
    lines = [f"Warehouse {config.WAREHOUSE_ID} on_hand for {sku}: {on_hand if row else 'no record'}."]
    if latest:
        lines.append(f"Latest warehouse receipt {latest['receipt_id']} ({latest['date']}): received "
                     f"{latest['received_qty']} of {latest['po_qty']} ordered. Note: {latest['damage_notes']}.")
    facts = {"sku": sku, "high_demand": is_high_demand(sku), "warehouse_on_hand": on_hand,
             "latest_warehouse_receipt": latest}
    return DataResult(tool="warehouse_status", facts=facts, findings=["warehouse_depleted"] if depleted else [],
                      summary_lines=lines, data_used=[INVENTORY_CSV, RECEIPTS_CSV])


# Vendor id -> fill and on-time rates; below threshold is a warning signal, not proof (04#vendor-performance).
def vendor_performance(vendor_id: str) -> DataResult:
    vendors = load_table("vendor_master")
    row = vendors[vendors["vendor_id"] == vendor_id].iloc[0]
    fill_rate, on_time_rate = float(row["fill_rate"]), float(row["on_time_rate"])
    warning = fill_rate < config.VENDOR_FILL_WARN or on_time_rate < config.VENDOR_ONTIME_WARN
    line = (f"Vendor {vendor_id} {row['name']}: fill rate {fill_rate:.0%}, on-time rate {on_time_rate:.0%}, "
            f"status {row['status']}.")
    if warning:
        line += " Below the 80% fill / 85% on-time thresholds: a warning signal for review, not proof of fault."
    facts = {"vendor_id": vendor_id, "vendor_name": row["name"], "fill_rate": fill_rate,
             "on_time_rate": on_time_rate, "status": row["status"], "warning": warning}
    return DataResult(tool="vendor_performance", facts=facts, findings=["vendor_warning"] if warning else [],
                      summary_lines=[line], data_used=[VENDOR_CSV])


# Store + SKU -> support cases for the same SKU in the window (evidence only, no findings).
# Same SKU at any store, because a vendor-side cause shows up across stores; same-store count reported too.
def related_cases(store_id: str, sku: str, days: int = config.RELATED_CASES_DAYS) -> DataResult:
    cases = within_window(load_table("support_cases"), "created_ts", days=days)
    rows = cases[cases["sku"] == sku].sort_values("created_ts", ascending=False)
    same_store = rows[rows["store_id"] == store_id]
    open_count = int(same_store["status"].isin(OPEN_STATUSES).sum())
    hints = merge_unique([same_store["root_cause_hint"].tolist()])[:config.MAX_LIST_ITEMS]
    evidence = same_store[["case_id", "store_id", "category_label", "priority_label", "status", "root_cause_hint"]]
    facts = {"store_id": store_id, "sku": sku, "days": days, "case_count_all_stores": len(rows),
             "case_count_this_store": len(same_store), "open_count_this_store": open_count, "root_cause_hints": hints}
    line = (f"{len(same_store)} support cases for {sku} at {store_id} in the last {days} days ({open_count} open); "
            f"{len(rows)} across all stores. Recorded hints: {'; '.join(hints) or 'none'}.")
    return DataResult(tool="related_cases", facts=facts, findings=[], summary_lines=[line], data_used=[CASES_CSV],
                      evidence_rows=evidence.head(config.MAX_LIST_ITEMS).to_dict("records"))


# Store id -> its nearby_store_id from store_master, or None for unknown stores / the warehouse.
def nearby_store(store_id: str) -> str | None:
    stores = load_table("store_master")
    rows = stores[stores["store_id"] == store_id]
    return rows["nearby_store_id"].iloc[0] if len(rows) else None


# Store + SKU -> other stores holding stock above their reorder point (nearby store first).
# Only a recommendation for manager-approved review (03#transfer-and-reorder-recommendation).
def transfer_options(store_id: str, sku: str) -> DataResult:
    inventory = load_table("inventory")
    nearby = nearby_store(store_id)
    has_spare = inventory["on_hand"] > inventory["reorder_point"]
    other_store = ~inventory["store_id"].isin([store_id, config.WAREHOUSE_ID])
    rows = inventory[(inventory["sku"] == sku) & has_spare & other_store]
    rows = rows.assign(is_nearby=rows["store_id"] == nearby).sort_values(["is_nearby", "on_hand"], ascending=False)
    options = []
    for row in rows.head(config.MAX_LIST_ITEMS).to_dict("records"):
        options.append({"store_id": row["store_id"], "on_hand": int(row["on_hand"]),
                        "reorder_point": int(row["reorder_point"]), "is_nearby": bool(row["is_nearby"])})
    lines = []
    for option in options[:2]:
        nearby_label = " (nearby)" if option["is_nearby"] else ""
        lines.append(f"{option['store_id']}{nearby_label} has {option['on_hand']} on hand "
                     f"(reorder point {option['reorder_point']}); transfer needs manager-approved review.")
    if not options:
        lines.append(f"No other store holds {sku} above its reorder point.")
    facts = {"store_id": store_id, "sku": sku, "nearby_store_id": nearby, "options": options}
    return DataResult(tool="transfer_options", facts=facts, findings=["transfer_possible"] if options else [],
                      summary_lines=lines, data_used=[INVENTORY_CSV, STORES_CSV])


# Finding code -> likely-cause sentence, in the policy's root-cause check order (03#root-cause-hints).
CAUSE_TEXT = [
    ("inventory_discrepancy", "System quantity and on-hand count differ (possible false availability); "
                              "a cycle count is recommended."),
    ("consecutive_short_shipment", "The last two receipts were short-shipped (below 90% of the PO)."),
    ("short_shipment", "A recent receipt was below 90% of the ordered quantity."),
    ("receipt_quality_issue", "A recent receipt failed its quality check."),
    ("warehouse_depleted", "Warehouse stock is depleted, so replenishment cannot ship from the warehouse."),
    ("vendor_warning", "Vendor fill / on-time rate is below threshold (a warning signal, not proof)."),
]


# Finding codes -> ranked likely causes (short_shipment is skipped when the consecutive version is present).
def likely_causes(findings: list[str]) -> list[str]:
    causes = []
    for code, sentence in CAUSE_TEXT:
        if code == "short_shipment" and "consecutive_short_shipment" in findings:
            continue
        if code in findings:
            causes.append(sentence)
    return causes


# Store + SKU -> every check in policy order (03#root-cause-hints), merged into one result.
def stockout_root_cause(store_id: str, sku: str) -> DataResult:
    product = product_row(sku)
    steps = [stock_status(store_id, sku), receipt_history(store_id, sku), warehouse_status(sku),
             vendor_performance(product["vendor_id"]), related_cases(store_id, sku), transfer_options(store_id, sku)]
    findings = merge_unique([step.findings for step in steps])
    facts = {"store_id": store_id, "sku": sku, "product": product["name"],
             "high_demand": bool(product["high_demand"]), "likely_causes": likely_causes(findings)}
    for step in steps:
        facts[step.tool] = step.facts
    return DataResult(tool="stockout_root_cause", facts=facts, findings=findings,
                      summary_lines=merge_unique([step.summary_lines for step in steps]),
                      data_used=merge_unique([step.data_used for step in steps]),
                      evidence_rows=merge_unique([step.evidence_rows for step in steps]))


# --- Quality tools: 04_vendor_quality_policy.md#product-quality-complaints ---------------------------

# Window -> one row per quality complaint (returns_quality reasons/notes + quality support cases),
# joined to product name and vendor. "Damaged packaging" returns are not product-quality complaints.
def quality_complaint_rows(days: int) -> pd.DataFrame:
    returns = within_window(load_table("returns_quality"), "created_ts", days=days)
    returns_text = returns["reason"] + " " + returns["notes"]
    is_quality = returns_text.str.contains(QUALITY_PATTERN, case=False)
    is_packaging = returns["reason"].str.contains("damaged packaging", case=False)
    from_returns = pd.DataFrame({"source_id": returns["return_id"], "store_id": returns["store_id"],
                                 "sku": returns["sku"], "text": returns_text})[is_quality & ~is_packaging]
    cases = within_window(load_table("support_cases"), "created_ts", days=days)
    cases = cases[cases["category_label"] == "Product Quality and Safety"]
    from_cases = pd.DataFrame({"source_id": cases["case_id"], "store_id": cases["store_id"], "sku": cases["sku"],
                               "text": cases["issue_text"] + " " + cases["root_cause_hint"]})
    rows = pd.concat([from_returns, from_cases], ignore_index=True)
    products = load_table("product_master")[["sku", "name", "vendor_id"]]
    rows = rows.merge(products, on="sku", how="left")
    return rows.fillna({"name": "", "vendor_id": ""})  # cases without a SKU have no product


# Complaint rows -> rows for one SKU, one vendor, or whose product name / text has all mention words.
def filter_complaints(rows: pd.DataFrame, sku: str | None, vendor_id: str | None,
                      mention: str | None) -> pd.DataFrame:
    if sku:
        rows = rows[rows["sku"] == sku]
    if vendor_id:
        rows = rows[rows["vendor_id"] == vendor_id]
    if mention:
        words = name_words(mention)
        rows = rows[(rows["name"] + " " + rows["text"]).apply(lambda text: words <= name_words(text))]
    return rows


# SKU / vendor / product mention -> complaint counts by store and SKU in the window, with findings.
def quality_complaints(sku: str | None = None, vendor_id: str | None = None, mention: str | None = None,
                       days: int = config.QUALITY_WINDOW_DAYS) -> DataResult:
    rows = filter_complaints(quality_complaint_rows(days), sku, vendor_id, mention)
    count, store_count = len(rows), int(rows["store_id"].nunique())
    findings = []
    if count >= config.REPEATED_COMPLAINTS_MIN or store_count >= config.REPEATED_COMPLAINTS_MIN:
        findings.append("repeated_quality_complaints")
    # Only for a specific product/vendor: one safety row somewhere in an all-products scan is not a live
    # incident for this request (same reason as vendor_quality_ranking).
    is_filtered = bool(sku or vendor_id or mention)
    if is_filtered and rows["text"].str.contains(SAFETY_ROW_PATTERN, case=False).any():
        findings.append("safety_incident")
    scope = sku or vendor_id or (f"'{mention}'" if mention else "all products")
    facts = {"filter": scope, "days": days, "complaint_count": count, "store_count": store_count,
             "by_store": count_by(rows, "store_id"), "by_sku": count_by(rows, "sku")}
    if sku:
        facts["high_demand"] = is_high_demand(sku)
    line = (f"{count} quality complaints for {scope} in the last {days} days across {store_count} stores "
            f"(by store: {facts['by_store']}).")
    evidence = rows[["source_id", "store_id", "sku"]].head(config.MAX_LIST_ITEMS).to_dict("records")
    return DataResult(tool="quality_complaints", facts=facts, findings=findings, summary_lines=[line],
                      data_used=[RETURNS_CSV, CASES_CSV, PRODUCT_CSV], evidence_rows=evidence)


# Ranked complaint table -> list of JSON-safe vendor dicts (top MAX_LIST_ITEMS).
def ranking_records(ranked: pd.DataFrame) -> list[dict]:
    ranking = []
    for row in ranked.head(config.MAX_LIST_ITEMS).to_dict("records"):
        ranking.append({"vendor_id": row["vendor_id"], "vendor_name": row["name"],
                        "complaint_count": int(row["complaint_count"]), "store_count": int(row["store_count"]),
                        "fill_rate": float(row["fill_rate"]), "on_time_rate": float(row["on_time_rate"])})
    return ranking


# Window -> vendors ranked by quality complaint count, with their fill / on-time rates.
# No safety_incident here: one safety row among all vendors must not make a ranking question P1.
def vendor_quality_ranking(days: int = config.QUALITY_WINDOW_DAYS) -> DataResult:
    rows = quality_complaint_rows(days)
    rows = rows[rows["vendor_id"] != ""]
    grouped = rows.groupby("vendor_id").agg(complaint_count=("source_id", "count"),
                                            store_count=("store_id", "nunique")).reset_index()
    vendors = load_table("vendor_master")[["vendor_id", "name", "fill_rate", "on_time_rate"]]
    ranked = grouped.merge(vendors, on="vendor_id").sort_values(
        ["complaint_count", "store_count", "vendor_id"], ascending=[False, False, True])
    ranking = ranking_records(ranked)
    top_is_repeated = bool(ranking) and ranking[0]["complaint_count"] >= config.REPEATED_COMPLAINTS_MIN
    lines = []
    for vendor in ranking[:3]:
        lines.append(f"{vendor['vendor_id']} {vendor['vendor_name']}: {vendor['complaint_count']} complaints "
                     f"across {vendor['store_count']} stores in the last {days} days.")
    lines.append("Complaint counts are a signal for vendor/product review, not proof of fault.")
    return DataResult(tool="vendor_quality_ranking", facts={"days": days, "ranking": ranking},
                      findings=["repeated_quality_complaints"] if top_is_repeated else [], summary_lines=lines,
                      data_used=[RETURNS_CSV, CASES_CSV, PRODUCT_CSV, VENDOR_CSV], evidence_rows=ranking)


# --- Analytics tools: 01_retail_operations_sop.md#analytics-rule ------------------------------------

# Category + priorities asked about -> escalation_matrix team for that category (matching priority first).
def category_team(category: str, priorities: list[str]) -> str:
    matrix = load_table("escalation_matrix")
    rows = matrix[matrix["category"] == category]
    preferred = rows[rows["priority"].isin(priorities)]
    if len(preferred) > 0:
        rows = preferred
    return rows["team"].iloc[0] if len(rows) else "Operations Analyst"


# Priorities + statuses + window -> cases in (ANCHOR - hours, ANCHOR] grouped by store and category.
def case_query(priorities: list[str], statuses: list[str] = OPEN_STATUSES,
               hours: int = config.CASE_QUERY_DEFAULT_HOURS) -> DataResult:
    cases = within_window(load_table("support_cases"), "created_ts", hours=hours)
    rows = cases[cases["priority_label"].isin(priorities) & cases["status"].isin(statuses)]
    groups = []
    for (store_id, category), group in rows.groupby(["store_id", "category_label"]):
        groups.append({"store_id": store_id, "category": category, "case_count": len(group),
                       "case_ids": group["case_id"].tolist(), "statuses": count_by(group, "status"),
                       "team": category_team(category, priorities)})
    by_store = count_by(rows, "store_id")
    lines = [f"{len(rows)} {'/'.join(priorities)} cases with status {' or '.join(statuses)} in the {hours} hours up "
             f"to {config.ANCHOR:%Y-%m-%d %H:%M} across {len(by_store)} stores."]
    for group in groups:
        lines.append(f"{group['store_id']} - {group['category']}: {group['case_count']} "
                     f"({', '.join(group['case_ids'])}) -> {group['team']}.")
    facts = {"priorities": priorities, "statuses": statuses, "window_hours": hours, "case_count": len(rows),
             "store_count": len(by_store), "by_store": by_store, "groups": groups}
    return DataResult(tool="case_query", facts=facts, findings=[], summary_lines=lines,
                      data_used=[CASES_CSV, MATRIX_CSV])


# Window -> receipts below 90% fill on or before the anchor, counted by vendor (evidence only).
def short_shipment_summary(days: int = config.QUALITY_WINDOW_DAYS) -> DataResult:
    receipts = within_window(load_table("receipts"), "receipt_date", days=days)
    short = receipts[receipts["received_qty"] < config.SHORT_SHIPMENT_PCT * receipts["po_qty"]]
    by_vendor = count_by(short, "vendor_id")
    lines = [f"{len(short)} of {len(receipts)} receipts in the last {days} days were below 90% of the PO."]
    for vendor_id, count in list(by_vendor.items())[:config.MAX_LIST_ITEMS]:
        lines.append(f"{vendor_id}: {count} short receipts.")
    facts = {"days": days, "receipt_count": len(receipts), "short_count": len(short), "by_vendor": by_vendor,
             "by_store": count_by(short, "store_id")}
    return DataResult(tool="short_shipment_summary", facts=facts, findings=[], summary_lines=lines,
                      data_used=[RECEIPTS_CSV])


# Window -> returns counted by reason and status, with no-receipt count (evidence only).
def returns_summary(days: int = config.QUALITY_WINDOW_DAYS) -> DataResult:
    returns = within_window(load_table("returns_quality"), "created_ts", days=days)
    no_receipt = int((~returns["receipt_present"]).sum())
    total_refund = float(returns["refund_amount_inr"].sum())
    by_reason = count_by(returns, "reason")
    lines = [f"{len(returns)} returns in the last {days} days ({no_receipt} without receipt), "
             f"total refund value INR {total_refund:,.0f}.",
             f"By reason: {dict(list(by_reason.items())[:config.MAX_LIST_ITEMS])}."]
    facts = {"days": days, "return_count": len(returns), "no_receipt_count": no_receipt,
             "total_refund_inr": total_refund, "by_reason": by_reason, "by_status": count_by(returns, "status")}
    return DataResult(tool="returns_summary", facts=facts, findings=[], summary_lines=lines,
                      data_used=[RETURNS_CSV])


# --- Returns and pricing tools: 02_pos_pricing_returns_policy.md ------------------------------------

# Refund amount + receipt status -> approval findings (#returns-and-refunds). Policy thresholds, no CSV.
def refund_check(amount_inr: float, receipt_present: bool | None) -> DataResult:
    no_receipt = receipt_present is False
    needs_approval = amount_inr >= config.REFUND_APPROVAL_ANY or (
        no_receipt and amount_inr >= config.REFUND_APPROVAL_NO_RECEIPT)
    findings = []
    if needs_approval:
        findings.append("refund_needs_approval")
    if no_receipt:
        findings.append("no_receipt_review")
    receipt_text = "without receipt" if no_receipt else "with receipt" if receipt_present else "receipt status unknown"
    line = f"Refund INR {amount_inr:,.0f} {receipt_text}: "
    line += "store manager approval is required before any refund." if needs_approval else "within standard limits."
    if no_receipt:
        line += " No-receipt returns always need manager review."
    facts = {"amount_inr": amount_inr, "receipt_present": receipt_present, "needs_approval": needs_approval,
             "approval_threshold_inr": config.REFUND_APPROVAL_ANY,
             "no_receipt_threshold_inr": config.REFUND_APPROVAL_NO_RECEIPT}
    return DataResult(tool="refund_check", facts=facts, findings=findings, summary_lines=[line], data_used=[])


# SKU + prices quoted in the request -> list price; mismatch confirmed when one quoted price equals the
# list price and another differs (02#pricing-and-promotions: verify shelf label and POS price).
def price_lookup(sku: str, quoted_prices: list[float]) -> DataResult:
    product = product_row(sku)
    list_price = float(product["list_price"])
    other_prices = [price for price in quoted_prices if price != list_price]
    confirmed = list_price in quoted_prices and len(other_prices) > 0
    line = f"{product['name']} ({sku}) list price in product_master: INR {list_price:,.0f}."
    if confirmed:
        line += f" The quoted INR {other_prices[0]:,.0f} does not match the list price."
    facts = {"sku": sku, "product": product["name"], "list_price_inr": list_price,
             "quoted_prices_inr": quoted_prices, "high_demand": bool(product["high_demand"])}
    return DataResult(tool="price_lookup", facts=facts, findings=["price_mismatch_confirmed"] if confirmed else [],
                      summary_lines=[line], data_used=[PRODUCT_CSV])


# --- Dispatcher -------------------------------------------------------------------------------------

# Time window in hours (or None) -> whole days for 30-day style tools (at least 1).
def window_days(hours: int | None) -> int:
    if hours is None:
        return config.QUALITY_WINDOW_DAYS
    return max(1, hours // 24)


# Analytics triage -> the one summary tool for its intent (deterministic, no LLM-written pandas).
def run_analytics(triage: TriageResult) -> list[DataResult]:
    facts = triage.facts
    intent = triage.analytics_intent
    days = window_days(facts.time_window_hours)
    if intent == "open_cases_window":
        priorities = facts.priority_filter or config.CASE_QUERY_PRIORITIES
        return [case_query(priorities, hours=facts.time_window_hours or config.CASE_QUERY_DEFAULT_HOURS)]
    if intent == "vendor_quality_ranking":
        return [vendor_quality_ranking(days)]
    if intent == "quality_trend":
        return [quality_complaints(sku=facts.sku, mention=None if facts.sku else facts.product_mention, days=days)]
    if intent == "short_shipments":
        return [short_shipment_summary(days)]
    if intent == "returns_summary":
        return [returns_summary(days)]
    if intent == "stockout_root_cause" and facts.store_id and facts.sku:
        return [stockout_root_cause(facts.store_id, facts.sku)]
    return []


# Returns facts -> refund_check when the amount is known, plus quality_complaints when the item is
# damaged/defective (02_pos_pricing_returns_policy.md#returns-and-refunds: link damaged returns to quality).
def returns_tools(facts: Facts) -> list[DataResult]:
    results = []
    if facts.amount_inr is not None:
        results.append(refund_check(facts.amount_inr, facts.receipt_present))
    if facts.defect_mentioned and (facts.sku or facts.product_mention):
        results.append(quality_complaints(sku=facts.sku, mention=None if facts.sku else facts.product_mention))
    return results


# Triage -> data tool results (ARCHITECTURE.md §6 [3B] dispatcher). Missing store/SKU -> no data;
# Triage missing_fields then drives the follow-up question.
def run_data_agent(triage: TriageResult) -> list[DataResult]:
    facts = triage.facts
    if triage.mode == "analytics":
        return run_analytics(triage)
    if triage.category == "Inventory and Replenishment" and facts.store_id and facts.sku:
        return [stockout_root_cause(facts.store_id, facts.sku)]
    if triage.category == "Vendor and Supply Chain" and facts.store_id and facts.sku:
        return [receipt_history(facts.store_id, facts.sku), vendor_performance(product_row(facts.sku)["vendor_id"])]
    if triage.category == "Product Quality and Safety" and (facts.sku or facts.product_mention):
        return [quality_complaints(sku=facts.sku, mention=None if facts.sku else facts.product_mention)]
    if triage.category == "Returns and Refunds":
        return returns_tools(facts)
    if triage.category == "Pricing and Promotions" and facts.sku:
        return [price_lookup(facts.sku, facts.quoted_prices_inr)]
    return []
