/**
 * This is a helper script to copy landing.jpg to the correct location
 * Run with: node copy-landing-image.js
 */
const fs = require('fs');
const path = require('path');

// Source and destination paths
const sourcePath = path.join(__dirname, '..', 'frontend', 'public', 'landing.jpg');
const destPath = path.join(__dirname, 'public', 'landing.jpg');

try {
  // Check if source exists
  if (fs.existsSync(sourcePath)) {
    // Copy the file
    fs.copyFileSync(sourcePath, destPath);
    console.log('✅ Successfully copied landing.jpg to public folder');
  } else {
    console.error('❌ Source file not found at:', sourcePath);
    console.log('Please manually copy your landing.jpg file to:', destPath);
  }
} catch (err) {
  console.error('Error copying file:', err);
}
