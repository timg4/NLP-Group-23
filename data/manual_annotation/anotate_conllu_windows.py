#!/usr/bin/env python3
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple


TAGS = ["O", "B-ORG", "I-ORG", "B-MON", "I-MON", "B-LEG", "I-LEG"]
AUTOSAVE_EVERY_N_LABELS = 25  # adjust if desired


@dataclass
class TokenLine:
    cols: List[str]
    original_line: str


@dataclass
class SentenceBlock:
    lines: List[Union[str, TokenLine]] = field(default_factory=list)

    def tokens(self) -> List[TokenLine]:
        return [x for x in self.lines if isinstance(x, TokenLine)]

    def get_text_comment(self) -> Optional[str]:
        for obj in self.lines:
            if isinstance(obj, str) and obj.startswith("# text"):
                parts = obj.split("=", 1)
                return parts[-1].strip() if len(parts) == 2 else obj
        return None


def is_token_line(line: str) -> bool:
    if line.startswith("#"):
        return False
    if "\t" not in line:
        return False
    cols = line.split("\t")
    if len(cols) < 11:
        return False
    tid = cols[0]
    if "-" in tid or "." in tid:
        return False
    return tid.isdigit()


def parse_conllu(path: Path) -> List[SentenceBlock]:
    blocks: List[SentenceBlock] = []
    current = SentenceBlock()

    def flush():
        nonlocal current
        if current.lines:
            blocks.append(current)
        current = SentenceBlock()

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.strip() == "":
            flush()
            continue
        if is_token_line(line):
            cols = line.split("\t")
            current.lines.append(TokenLine(cols=cols, original_line=line))
        else:
            current.lines.append(line)

    flush()
    return blocks


def write_conllu(blocks: List[SentenceBlock], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for b in blocks:
            for obj in b.lines:
                if isinstance(obj, TokenLine):
                    f.write("\t".join(obj.cols) + "\n")
                else:
                    f.write(obj + "\n")
            f.write("\n")


def token_text_and_spans(tokens: List[TokenLine]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Reconstruct sentence text from tokens and return character spans per token
    using MISC SpaceAfter=No to control spacing.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0

    for t in tokens:
        form = t.cols[1] if len(t.cols) > 1 else ""
        start = cursor
        parts.append(form)
        cursor += len(form)
        end = cursor
        spans.append((start, end))

        misc = t.cols[9] if len(t.cols) > 9 else ""
        if "SpaceAfter=No" not in (misc or ""):
            parts.append(" ")
            cursor += 1

    text = "".join(parts).rstrip()
    return text, spans


def tag_type(tag: str) -> Optional[str]:
    if tag.endswith("ORG"):
        return "ORG"
    if tag.endswith("MON"):
        return "MON"
    if tag.endswith("LEG"):
        return "LEG"
    return None


def bio_inconsistencies(tags: List[str]) -> List[bool]:
    errs = [False] * len(tags)
    prev = "O"
    for i, t in enumerate(tags):
        if t.startswith("I-"):
            x = t[2:]
            if not (prev == f"B-{x}" or prev == f"I-{x}"):
                errs[i] = True
        prev = t
    return errs


def fix_bio_minimal(tags: List[str]) -> List[str]:
    fixed = tags[:]
    prev = "O"
    for i, t in enumerate(fixed):
        if t.startswith("I-"):
            x = t[2:]
            if not (prev == f"B-{x}" or prev == f"I-{x}"):
                fixed[i] = f"B-{x}"
        prev = fixed[i]
    return fixed


class ReviewWindow(tk.Toplevel):
    def __init__(self, parent, sentence_block: SentenceBlock, on_set_tag, on_confirm, on_back, on_mark_sentence_o):
        super().__init__(parent)
        self.title("Review sentence")
        self.geometry("980x560")
        self.resizable(True, True)

        self.sentence_block = sentence_block
        self.tokens = sentence_block.tokens()
        self.on_set_tag = on_set_tag
        self.on_confirm = on_confirm
        self.on_back = on_back
        self.on_mark_sentence_o = on_mark_sentence_o

        self._build_ui()
        self._populate()
        self._refresh_bio_status()

        for i, tag in enumerate(TAGS, start=1):
            self.bind(str(i), lambda e, t=tag: self._set_tag_selected(t))

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        text, _ = token_text_and_spans(self.tokens)

        header = ttk.Frame(frm)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(header, text="Sentence (review):", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(header, text="Mark whole sentence as O", command=self._mark_sentence_o).pack(side="right", padx=6)

        self.text_box = tk.Text(frm, height=4, wrap="word")
        self.text_box.insert("1.0", text)
        self.text_box.configure(state="disabled")
        self.text_box.pack(fill="x", pady=(0, 10))

        cols = ("id", "form", "tag", "bio")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", height=14)
        self.tree.heading("id", text="ID")
        self.tree.heading("form", text="FORM")
        self.tree.heading("tag", text="TAG")
        self.tree.heading("bio", text="BIO")
        self.tree.column("id", width=60, anchor="center")
        self.tree.column("form", width=420, anchor="w")
        self.tree.column("tag", width=140, anchor="center")
        self.tree.column("bio", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True)

        self.tree.tag_configure("bio_error", background="#ffe5e5")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._update_selected_label())

        self.status_line = ttk.Label(frm, text="")
        self.status_line.pack(anchor="w", pady=(8, 4))

        self.selected_info = ttk.Label(frm, text="Selected: (none)")
        self.selected_info.pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(0, 10))
        for tag in TAGS:
            ttk.Button(btns, text=tag, command=lambda t=tag: self._set_tag_selected(t)).pack(side="left", padx=4, pady=4)

        actions = ttk.Frame(frm)
        actions.pack(fill="x")

        ttk.Button(actions, text="Back to token view", command=self._back).pack(side="left")
        ttk.Button(actions, text="Validate BIO", command=self._refresh_bio_status).pack(side="left", padx=8)
        ttk.Button(actions, text="Fix BIO (minimal)", command=self._fix_bio).pack(side="left")
        ttk.Button(actions, text="Confirm sentence", command=self._confirm).pack(side="right")

    def _populate(self):
        self.tree.delete(*self.tree.get_children())
        for i, tok in enumerate(self.tokens):
            tid = tok.cols[0]
            form = tok.cols[1] if len(tok.cols) > 1 else ""
            tag = tok.cols[-1] if tok.cols else ""
            self.tree.insert("", "end", iid=str(i), values=(tid, form, tag, ""))

        if self.tokens:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self._update_selected_label()

    def _update_selected_label(self):
        sel = self.tree.selection()
        if not sel:
            self.selected_info.config(text="Selected: (none)")
            return
        idx = int(sel[0])
        tok = self.tokens[idx]
        self.selected_info.config(text=f"Selected: id={tok.cols[0]} | form={tok.cols[1]} | tag={tok.cols[-1]}")

    def _set_tag_selected(self, tag: str):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a token row first.")
            return
        idx = int(sel[0])
        self.on_set_tag(idx, tag)
        tok = self.tokens[idx]
        self.tree.item(str(idx), values=(tok.cols[0], tok.cols[1], tok.cols[-1], ""))
        self._refresh_bio_status()
        self._update_selected_label()

    def _refresh_bio_status(self):
        tags = [t.cols[-1] for t in self.tokens]
        errs = bio_inconsistencies(tags)

        any_err = any(errs)
        for i, is_err in enumerate(errs):
            tok = self.tokens[i]
            bio_txt = "ERR" if is_err else "OK"
            row_tags = ("bio_error",) if is_err else ()
            self.tree.item(str(i), values=(tok.cols[0], tok.cols[1], tok.cols[-1], bio_txt), tags=row_tags)

        self.status_line.config(text="BIO validation: inconsistencies found (rows highlighted)." if any_err else "BIO validation: OK.")

    def _fix_bio(self):
        tags = [t.cols[-1] for t in self.tokens]
        fixed = fix_bio_minimal(tags)
        changed = 0
        for i, new_tag in enumerate(fixed):
            if new_tag != self.tokens[i].cols[-1]:
                self.on_set_tag(i, new_tag)
                changed += 1
        self._populate()
        self._refresh_bio_status()
        messagebox.showinfo("Fix BIO", f"Applied minimal BIO fix. Changes: {changed}")

    def _confirm(self):
        if any(t.cols and t.cols[-1] == "_" for t in self.tokens):
            messagebox.showwarning("Incomplete", "This sentence still contains '_' tags. Please label all tokens.")
            return

        tags = [t.cols[-1] for t in self.tokens]
        if any(bio_inconsistencies(tags)):
            proceed = messagebox.askyesno("BIO warning", "BIO inconsistencies are still present.\n\nConfirm sentence anyway?")
            if not proceed:
                return

        self.on_confirm()
        self.destroy()

    def _back(self):
        self.on_back()
        self.destroy()

    def _mark_sentence_o(self):
        self.on_mark_sentence_o()
        self._populate()
        self._refresh_bio_status()


class App:
    def __init__(self, root: tk.Tk, blocks: List[SentenceBlock], input_path: Path, output_path: Path):
        self.root = root
        self.blocks = blocks
        self.input_path = input_path
        self.output_path = output_path
        self.autosave_path = output_path.with_name(output_path.stem + ".autosave" + output_path.suffix)

        self.token_sentence_indices = [i for i, b in enumerate(blocks) if b.tokens()]
        if not self.token_sentence_indices:
            raise SystemExit("No token sentences found in input file.")

        self.pos_in_token_sents = 0
        self.token_idx = 0
        self.confirmed = set()
        self.label_actions = 0

        self._build_ui()
        self._render()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.title("CoNLL-U Manual NER Annotation (Windows)")
        self.root.geometry("1050x720")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 8))

        self.lbl_progress = ttk.Label(top, text="", font=("TkDefaultFont", 11, "bold"))
        self.lbl_progress.pack(side="left")

        ttk.Button(top, text="Save now", command=self.save).pack(side="right")
        ttk.Button(top, text="Autosave now", command=self.autosave).pack(side="right", padx=8)

        sent_frame = ttk.LabelFrame(main, text="Sentence (current token is highlighted)")
        sent_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.txt_sentence = tk.Text(sent_frame, wrap="word", height=10)
        self.txt_sentence.configure(state="disabled")
        yscroll = ttk.Scrollbar(sent_frame, command=self.txt_sentence.yview)
        self.txt_sentence.configure(yscrollcommand=yscroll.set)

        self.txt_sentence.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.txt_sentence.tag_configure("CURRENT", background="#fff2a8")
        self.txt_sentence.tag_configure("ORG", background="#e7f0ff")
        self.txt_sentence.tag_configure("MON", background="#e9ffe7")
        self.txt_sentence.tag_configure("LEG", background="#fff3e0")

        tok_frame = ttk.LabelFrame(main, text="Current token")
        tok_frame.pack(fill="x", pady=(0, 10))

        self.lbl_token_big = ttk.Label(tok_frame, text="", font=("TkDefaultFont", 16, "bold"))
        self.lbl_token_big.pack(anchor="w", padx=8, pady=(10, 2))

        self.lbl_token_meta = ttk.Label(tok_frame, text="", font=("TkDefaultFont", 11))
        self.lbl_token_meta.pack(anchor="w", padx=8, pady=(0, 10))

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(0, 8))
        for tag in TAGS:
            ttk.Button(btns, text=tag, command=lambda t=tag: self.set_tag(t)).pack(side="left", padx=4, pady=4)

        quick = ttk.Frame(main)
        quick.pack(fill="x", pady=(0, 10))

        self.btn_o_sentence = ttk.Button(quick, text="", command=self.mark_sentence_or_rest_o)
        self.btn_o_sentence.pack(side="left", padx=4)

        ttk.Button(quick, text="Review/Confirm this sentence", command=self.open_review).pack(side="left", padx=4)

        nav = ttk.Frame(main)
        nav.pack(fill="x")

        ttk.Button(nav, text="<< Prev token", command=self.prev_token).pack(side="left", padx=4)
        ttk.Button(nav, text="Next token >>", command=self.next_token).pack(side="left", padx=4)

        ttk.Separator(nav, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(nav, text="<< Prev sentence", command=self.prev_sentence).pack(side="left", padx=4)
        ttk.Button(nav, text="Next sentence >>", command=self.next_sentence).pack(side="left", padx=4)

        ttk.Separator(nav, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(nav, text="Go to first untagged (_)", command=self.goto_first_underscore).pack(side="left", padx=4)

        for i, tag in enumerate(TAGS, start=1):
            self.root.bind(str(i), lambda e, t=tag: self.set_tag(t))

        self.root.bind("<Left>", lambda e: self.prev_token())
        self.root.bind("<Right>", lambda e: self.next_token())
        self.root.bind("<Up>", lambda e: self.prev_sentence())
        self.root.bind("<Down>", lambda e: self.next_sentence())
        self.root.bind("<space>", lambda e: self.next_token())
        self.root.bind("<BackSpace>", lambda e: self.prev_token())

    def current_block_index(self) -> int:
        return self.token_sentence_indices[self.pos_in_token_sents]

    def current_block(self) -> SentenceBlock:
        return self.blocks[self.current_block_index()]

    def current_tokens(self) -> List[TokenLine]:
        return self.current_block().tokens()

    def current_token(self) -> TokenLine:
        return self.current_tokens()[self.token_idx]

    def sentence_has_any_label(self) -> bool:
        # "token eingetragen" means not '_' (O or B/I-*)
        return any(t.cols[-1] != "_" for t in self.current_tokens())

    def update_o_button_label(self):
        if self.sentence_has_any_label():
            self.btn_o_sentence.config(text="Rest des Satzes auf O")
        else:
            self.btn_o_sentence.config(text="Ganzen Satz auf O")

    def _render(self):
        sent_pos = self.pos_in_token_sents + 1
        sent_total = len(self.token_sentence_indices)
        toks = self.current_tokens()

        confirmed_flag = "CONFIRMED" if self.current_block_index() in self.confirmed else "NOT CONFIRMED"
        self.lbl_progress.config(
            text=f"Sentence {sent_pos}/{sent_total} ({confirmed_flag}) | Token {self.token_idx + 1}/{len(toks)} "
                 f"| Output: {self.output_path.name} | Autosave: every {AUTOSAVE_EVERY_N_LABELS} labels"
        )

        text, spans = token_text_and_spans(toks)

        self.txt_sentence.configure(state="normal")
        self.txt_sentence.delete("1.0", "end")
        self.txt_sentence.insert("1.0", text)

        for tg in ["CURRENT", "ORG", "MON", "LEG"]:
            self.txt_sentence.tag_remove(tg, "1.0", "end")

        for i, tok in enumerate(toks):
            tag = tok.cols[-1]
            ttype = tag_type(tag) if tag else None
            if ttype in {"ORG", "MON", "LEG"}:
                start, end = spans[i]
                self.txt_sentence.tag_add(ttype, f"1.0+{start}c", f"1.0+{end}c")

        cur_start, cur_end = spans[self.token_idx]
        self.txt_sentence.tag_add("CURRENT", f"1.0+{cur_start}c", f"1.0+{cur_end}c")
        self.txt_sentence.configure(state="disabled")

        tok = self.current_token()
        tid = tok.cols[0]
        form = tok.cols[1] if len(tok.cols) > 1 else ""
        cur_tag = tok.cols[-1]

        prev_form = toks[self.token_idx - 1].cols[1] if self.token_idx > 0 else ""
        next_form = toks[self.token_idx + 1].cols[1] if self.token_idx < len(toks) - 1 else ""
        context = f"{prev_form}   [{form}]   {next_form}".strip()

        self.lbl_token_big.config(text=f"{form}")
        self.lbl_token_meta.config(text=f"id={tid} | current tag={cur_tag} | context: {context}")

        self.update_o_button_label()

    def mark_dirty(self):
        bi = self.current_block_index()
        if bi in self.confirmed:
            self.confirmed.remove(bi)

    def maybe_autosave(self, force: bool = False):
        if force:
            write_conllu(self.blocks, self.autosave_path)
            return
        if AUTOSAVE_EVERY_N_LABELS > 0 and (self.label_actions % AUTOSAVE_EVERY_N_LABELS == 0):
            write_conllu(self.blocks, self.autosave_path)

    def set_tag(self, tag: str):
        tok = self.current_token()
        tok.cols[-1] = tag
        self.mark_dirty()

        self.label_actions += 1
        self.maybe_autosave()

        if self.token_idx == len(self.current_tokens()) - 1:
            self.open_review(auto_after_last_token=True)
        else:
            self.token_idx += 1
            self._render()

    def mark_sentence_or_rest_o(self):
        toks = self.current_tokens()
        self.mark_dirty()

        if not self.sentence_has_any_label():
            # whole sentence to O
            for t in toks:
                t.cols[-1] = "O"
        else:
            # remaining sentence (from current token to end) to O
            for i in range(self.token_idx, len(toks)):
                toks[i].cols[-1] = "O"

        self.label_actions += 1
        self.maybe_autosave()
        self.open_review()

    def open_review(self, auto_after_last_token: bool = False):
        bi = self.current_block_index()
        block = self.current_block()
        tokens = block.tokens()

        def on_set_tag(token_local_idx: int, tag: str):
            tokens[token_local_idx].cols[-1] = tag
            if bi in self.confirmed:
                self.confirmed.remove(bi)
            self.label_actions += 1
            self.maybe_autosave()

        def on_mark_sentence_o():
            for t in tokens:
                t.cols[-1] = "O"
            if bi in self.confirmed:
                self.confirmed.remove(bi)
            self.label_actions += 1
            self.maybe_autosave()

        def on_confirm():
            self.confirmed.add(bi)
            self.maybe_autosave(force=True)

            if not self._advance_sentence():
                self._render()
                messagebox.showinfo("Done", "All sentences completed. Saving final output now.")
                self.save()
                return
            self._render()

        def on_back():
            self._render()

        ReviewWindow(
            self.root,
            block,
            on_set_tag=on_set_tag,
            on_confirm=on_confirm,
            on_back=on_back,
            on_mark_sentence_o=on_mark_sentence_o,
        )

    def _advance_sentence(self) -> bool:
        if self.pos_in_token_sents + 1 >= len(self.token_sentence_indices):
            return False
        self.pos_in_token_sents += 1
        self.token_idx = 0
        return True

    def _retreat_sentence(self) -> bool:
        if self.pos_in_token_sents == 0:
            return False
        self.pos_in_token_sents -= 1
        self.token_idx = 0
        return True

    def next_token(self):
        if self.token_idx < len(self.current_tokens()) - 1:
            self.token_idx += 1
            self._render()
            return
        self.open_review(auto_after_last_token=True)

    def prev_token(self):
        if self.token_idx > 0:
            self.token_idx -= 1
            self._render()
            return
        messagebox.showinfo("Start", "Already at the first token of this sentence.")

    def next_sentence(self):
        bi = self.current_block_index()
        if bi not in self.confirmed:
            self.open_review()
            return
        if not self._advance_sentence():
            messagebox.showinfo("End", "Already at the last sentence.")
            return
        self._render()

    def prev_sentence(self):
        if not self._retreat_sentence():
            messagebox.showinfo("Start", "Already at the first sentence.")
            return
        self._render()

    def goto_first_underscore(self):
        def find_from(start_sent_pos: int, start_tok: int):
            for sp in range(start_sent_pos, len(self.token_sentence_indices)):
                b = self.blocks[self.token_sentence_indices[sp]]
                toks = b.tokens()
                t0 = start_tok if sp == start_sent_pos else 0
                for ti in range(t0, len(toks)):
                    if toks[ti].cols[-1] == "_":
                        return sp, ti
            return None

        hit = find_from(self.pos_in_token_sents, self.token_idx)
        if hit is None:
            hit = find_from(0, 0)

        if hit is None:
            messagebox.showinfo("Info", "No '_' tags found. Everything seems annotated already.")
            return

        self.pos_in_token_sents, self.token_idx = hit
        self._render()

    def autosave(self):
        write_conllu(self.blocks, self.autosave_path)
        messagebox.showinfo("Autosaved", f"Autosaved to:\n{self.autosave_path.resolve()}")

    def save(self):
        write_conllu(self.blocks, self.output_path)
        messagebox.showinfo("Saved", f"Saved to:\n{self.output_path.resolve()}")

    def on_close(self):
        # Always autosave for safety
        write_conllu(self.blocks, self.autosave_path)

        if messagebox.askyesno("Quit", "Save final output before quitting?"):
            write_conllu(self.blocks, self.output_path)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Manual GUI annotator for CoNLL-U 11th column (Windows) - v3.")
    parser.add_argument("--input", required=True, help="Input .conllu file")
    parser.add_argument("--output", default="sentence_hand_labeled.conllu", help="Output .conllu file")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    blocks = parse_conllu(in_path)

    root = tk.Tk()
    App(root, blocks, in_path, out_path)
    root.mainloop()


if __name__ == "__main__":
    main()
