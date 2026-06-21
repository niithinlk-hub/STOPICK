import sharp from "sharp";
import { mkdir } from "node:fs/promises";

const LOGO = `
    <polygon points="32,6 54.52,19 54.52,45 32,58 9.48,45 9.48,19" stroke="#3B82F6" stroke-opacity="0.35" stroke-width="1.8"/>
    <polygon points="32,11 50.19,21.5 50.19,42.5 32,53 13.81,42.5 13.81,21.5" stroke="#2DD4BF" stroke-opacity="0.6" stroke-width="1.8"/>
    <polygon points="32,16 45.86,24 45.86,40 32,48 18.14,40 18.14,24" stroke="#3B82F6" stroke-width="2"/>
    <line x1="32" y1="17" x2="32" y2="47" stroke="url(#g)" stroke-width="2.4" stroke-linecap="round"/>
    <rect x="27.4" y="23" width="9.2" height="18" rx="2.2" fill="url(#g)" stroke="none"/>`;
const DEFS = `<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#34D399"/><stop offset="100%" stop-color="#2DD4A0"/></linearGradient></defs>`;
const ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">${DEFS}<rect width="64" height="64" rx="14" fill="#0A0E17"/><g fill="none" stroke-linejoin="round">${LOGO}</g></svg>`;
const MASK = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">${DEFS}<rect width="64" height="64" fill="#0A0E17"/><g transform="translate(32 32) scale(0.78) translate(-32 -32)" fill="none" stroke-linejoin="round">${LOGO}</g></svg>`;

await mkdir("public/icons", { recursive: true });
const out = (svg, size, file) => sharp(Buffer.from(svg)).resize(size, size).png().toFile(file);
await out(ICON, 192, "public/icons/icon-192.png");
await out(ICON, 512, "public/icons/icon-512.png");
await out(MASK, 192, "public/icons/maskable-192.png");
await out(MASK, 512, "public/icons/maskable-512.png");
await out(ICON, 180, "public/apple-touch-icon.png");
await out(ICON, 32, "public/favicon-32.png");
console.log("icons generated");
