# VoltOps Marketing Homepage

A premium, dark, scroll-story marketing homepage for a niche SaaS ERP product focused on **small to mid-sized electronics e-commerce operators**.

> **AI agents** (Claude Code, Cursor, Codex, Copilot, …): start with [`AGENTS.md`](./AGENTS.md) — it briefs you on the install flow, common commands, architecture, and known gotchas in one read.
>
> **Cricket prediction pipeline** users: see [`cricket_pipeline/README.md`](./cricket_pipeline/README.md) for the full guide.

## What is included

- Conversion-focused one-page site with sharp niche messaging.
- Scroll-driven storytelling flow: chaos → pain → control → growth → CTA.
- Lenis smooth-scrolling setup inspired by modern immersive sites.
- Sticky navigation + repeated CTA placements.
- Lead capture demo form section.
- Mobile-friendly layout with sticky mobile CTA.

## Files

- `index.html` — complete page structure and conversion copy.
- `styles.css` — dark premium visual system, layout, responsive behavior.
- `script.js` — Lenis setup, reveal-on-scroll, subtle parallax, lead form behavior.
- `serve-phone.sh` — helper script to serve the site so other devices on your Wi-Fi (like your phone) can open it.

## Run locally (desktop)

```bash
python3 -m http.server 4173
```

Then open:

- http://localhost:4173

## View it on your phone (same Wi-Fi)

### Option A: one command helper

```bash
./serve-phone.sh
```

This starts the server on `0.0.0.0` and prints the phone-friendly URL.

### Option B: manual

```bash
python3 -m http.server 4173 --bind 0.0.0.0
```

Then find your computer's local IP and open this on your phone:

- `http://<your-local-ip>:4173`

Examples:

- macOS/Linux: `hostname -I` (Linux) or `ipconfig getifaddr en0` (macOS)
- Windows: `ipconfig`


## Share publicly (works off your Wi-Fi)

If you want to open the site on your phone from any network, use a temporary Cloudflare tunnel:

```bash
./share-public.sh
```

This will:

1. start the local site on `127.0.0.1:4173`
2. create a secure public `https://...trycloudflare.com` URL
3. print that URL so you can open it on your phone

> Requires `cloudflared` installed locally.

You can also pass a custom port:

```bash
./share-public.sh 5000
```

## Notes

- The page uses Lenis via CDN: `https://unpkg.com/lenis@1.3.13/dist/lenis.min.js`
- Form submit is frontend-only placeholder behavior for demo/waitlist capture flow.


## Quality checks

```bash
./check.sh
```
