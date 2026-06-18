# TIBrain Chrome Extension

This extension opens the existing TIBrain frontend in Chrome's side panel. It does not change or fork the current `client` app, so login, dashboard navigation, Gmail connection, and backend calls keep using the deployed website logic.

## Local Install

1. Copy `.env.example` to `.env`.
2. Set `EXTENSION_FRONTEND_URL` in `.env`.
3. Run `npm run build` inside this folder to generate `config.js`.
4. Open Chrome and go to `chrome://extensions`.
5. Enable **Developer mode**.
6. Click **Load unpacked**.
7. Select this `extension` folder.
8. Open **Details > Extension options** and confirm:
   - Production: `https://your-vercel-app.vercel.app`
   - Local dev: `http://localhost:5173`
9. Click the extension icon to open the side panel.

## Environment

Chrome extensions cannot read `.env` files directly at runtime. The build step reads `extension/.env` and writes the safe public frontend URL into `config.js`.

```env
EXTENSION_FRONTEND_URL=https://your-vercel-app.vercel.app
```

## Notes

- The backend CORS config must include the Vercel frontend origin.
- If the deployed site sends `X-Frame-Options` or restrictive `frame-ancestors`, remove or adjust that header so the extension side panel can embed it.
- Keep Google OAuth settings aligned with the website origin because the website still owns the auth flow.
