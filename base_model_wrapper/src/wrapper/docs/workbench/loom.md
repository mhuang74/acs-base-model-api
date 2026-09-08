# Loom

A **loom** is a tree view for exploring what a base model might say. Instead of one linear completion, you generate several branches at a point, pick the one you like, keep going, and backtrack whenever you want — keeping the whole "multiverse" as a navigable tree. Per-token logprobs are shown inline so you can see how confident the model was at each step.

Open **Loom** from the top nav. It lists your saved looms — click **＋ New loom** to start one, or open a recent one. Each loom is its own saved object (private to you), so the tree is still there when you come back.

## 1. Set a root

Type a starting prompt in the text box and click **Set root (replaces tree)**. The root is your prompt verbatim — no generation happens yet. (Re-setting the root clears the current tree, so you can start over cleanly.)

## 2. Branch

Select a node, then click **Branch from here**. The controls above it:

- **n** — how many completions to generate at once. Each becomes a sibling branch under the selected node.
- **max tokens** and **temperature** — the usual sampling controls (base-model defaults: `temperature` 1.0).
- **logprobs** — how many alternatives to record per token. This drives the heatmap (below); set it to `0` to skip.
- **seed** — leave it `auto` for fresh sampling, or pin a number to make a branch reproducible.

One click sends a **single** request for `n` completions; each non-empty one is added as a child of the selected node and streams in live.

## 3. Navigate the tree

- The **sidebar** shows the whole tree. Click any node to select it; the centre panel shows the full text from the root down to that node.
- Switch to the **Graph** view (the *List / Graph* toggle above the tree) for a graphical map of the whole "multiverse": one box per node, branching left→right, with the current path highlighted. **Drag** to pan, **scroll** (or the +/− buttons) to zoom, and **click** a node to select it — the node panel and branch controls work exactly as in the list view. Your choice of view is remembered.
- **Branch from** any node to explore a new direction. To **backtrack**, just select an earlier node and branch again — the original branches stay put.
- Delete a node (and everything under it) with the delete control on the node.

## 4. Edit, split, and hand-write branches

Looms are meant to be driven at speed. With a node selected, the panel shows three actions (and matching hotkeys):

- **Edit** (`e`) — change a node's text in place. Click **Save edit** to keep it. Editing a node re-uses its stored context, so its children continue from your edited text. (If your edit changes the text, the per-token heatmap for that node is dropped — logprobs only line up with the text the model actually produced. Opening a node and saving *without* changing anything keeps the heatmap intact.)
- **Split at cursor** (`s`) — the signature loom move. Cut the selected node in two at the cursor: the node keeps the text before the cursor, and a new child holds the text after it. Any children the node already had are moved onto that new child, so every deeper branch keeps its exact context. Splitting is how you insert a fork *inside* an existing completion. Pressing **Split at cursor** (or `s`) opens the edit box so you can place the cursor exactly, then click **Split here** to cut at that point — the edited text and the split commit together in one step.
- **New sibling** (`i`) — add a hand-written alternative continuation next to the selected node (same parent), then type into it. Only works on non-root nodes — to start from a different seed, make a new loom.

**Save vs. new sibling — the convention:** editing **mutates the node in place** (matching the reference loom's "edit node"); it does not fork. To keep the original *and* an alternative, use **New sibling** (or generate more branches).

## Hotkeys

Press **?** any time for the full legend. While your cursor is in a text box, letter keys type normally — shortcuts only fire when no field is focused.

| Key | Action |
| --- | --- |
| `↑` / `↓` | Previous / next sibling |
| `←` | Go to parent |
| `→` | Go to first child |
| `g` | Generate branches from the selected node |
| `e` | Edit the node's text |
| `s` | Split the node at the cursor |
| `i` | Insert a new sibling |
| `Delete` | Delete the node and its subtree |
| `Esc` | Cancel an edit / close the legend |
| `?` | Toggle the shortcut legend |

## Export / import

Buttons sit on the top view bar:

- **Export JSON** — download the whole tree as a JSON file (topology + text + logprobs), for backup or sharing.
- **Export text** — download a plain-text transcript of the *current path* (root → selected node), i.e. the full continuation you're reading.
- **Import JSON** — pick a previously exported JSON file to load it into a **brand-new loom**. Node IDs are regenerated on import, so importing never touches or collides with an existing loom.

## 5. Read the logprobs

When a generated node is selected, its tokens are colour-coded by probability — **green = the model was confident, red = surprising**. **Click a token** to see the top-*k* alternatives it considered and their probabilities. (The root, being your own text, isn't coloured.)

## 6. Reproduce a branch

Pin a **seed** before branching, and the same model + prompt + seed will reproduce that exact continuation — handy for sharing a path or debugging a specific generation. Leave the seed on `auto` to let each branch sample freshly (so `n` branches diverge).
