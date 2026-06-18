const fs = require('fs');
const path = require('path');

const extensionDir = __dirname;
const envPath = path.join(extensionDir, '.env');
const outputPath = path.join(extensionDir, 'config.js');

const parseEnv = (source) => {
  const values = {};
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const separatorIndex = trimmed.indexOf('=');
    if (separatorIndex === -1) continue;
    const key = trimmed.slice(0, separatorIndex).trim();
    const rawValue = trimmed.slice(separatorIndex + 1).trim();
    values[key] = rawValue.replace(/^['"]|['"]$/g, '');
  }
  return values;
};

const env = fs.existsSync(envPath) ? parseEnv(fs.readFileSync(envPath, 'utf8')) : {};
const frontendUrl = String(env.EXTENSION_FRONTEND_URL || '').trim().replace(/\/+$/, '');

const configSource = `window.TIBRAIN_EXTENSION_CONFIG = {
  defaultFrontendUrl: ${JSON.stringify(frontendUrl)},
};
`;

fs.writeFileSync(outputPath, configSource);
console.log(`Generated ${path.relative(process.cwd(), outputPath)}`);
