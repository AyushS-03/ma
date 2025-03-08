// This script creates base64 placeholder images for the landing page.
// Run this file with Node.js to create the image files.

const fs = require('fs');
const path = require('path');

// Create basic SVG placeholders for landing page images
const createLandingHeroImage = () => {
  // A simple SVG with gradients and shapes to represent a medical interface
  const svgContent = `
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
  <!-- Background gradient -->
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f5f7fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#c3cfe2;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4285f4;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#34a853;stop-opacity:0.9" />
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="100%" height="100%" fill="url(#grad1)" />
  
  <!-- Doctor silhouette -->
  <ellipse cx="400" cy="400" rx="200" ry="250" fill="#ffffff" stroke="#4285f4" stroke-width="3" />
  <circle cx="400" cy="300" r="90" fill="#f1f3f4" stroke="#4285f4" stroke-width="2" />
  <rect x="300" y="400" width="200" height="250" rx="20" fill="#ffffff" stroke="#4285f4" stroke-width="2" />
  
  <!-- Medical display elements -->
  <rect x="650" y="200" width="400" height="300" rx="15" fill="#ffffff" stroke="#4285f4" stroke-width="3" />
  <rect x="670" y="230" width="360" height="30" rx="5" fill="#e8f0fe" />
  <rect x="670" y="270" width="360" height="2" fill="#4285f4" />
  
  <!-- Medical data representation -->
  <rect x="670" y="290" width="360" height="190" rx="5" fill="#f8f9fa" />
  <polyline points="690,380 720,340 750,400 780,320 810,370 840,350 870,390 900,330 930,350 960,320 990,380" 
           stroke="#4285f4" fill="none" stroke-width="3" />
  <polyline points="690,400 720,420 750,380 780,430 810,390 840,410 870,370 900,420 930,400 960,430 990,380" 
           stroke="#34a853" fill="none" stroke-width="3" />
  
  <!-- Medical icons -->
  <circle cx="700" cy="520" r="20" fill="#4285f4" />
  <text x="700" y="525" font-family="Arial" font-size="20" fill="white" text-anchor="middle">+</text>
  
  <circle cx="760" cy="520" r="20" fill="#34a853" />
  <text x="760" y="527" font-family="Arial" font-size="24" fill="white" text-anchor="middle">♡</text>
  
  <circle cx="820" cy="520" r="20" fill="#fbbc05" />
  <text x="820" y="525" font-family="Arial" font-size="18" fill="white" text-anchor="middle">𝍪</text>
  
  <circle cx="880" cy="520" r="20" fill="#ea4335" />
  <text x="880" y="525" font-family="Arial" font-size="20" fill="white" text-anchor="middle">✓</text>
  
  <text x="600" y="600" font-family="Arial" font-size="28" fill="#4285f4" font-weight="bold" text-anchor="middle">MedAssist AI</text>
  <text x="600" y="640" font-family="Arial" font-size="18" fill="#5f6368" text-anchor="middle">Your Personal Health Assistant</text>
</svg>
  `;
  
  return Buffer.from(svgContent);
}

const createDemoScreenshotImage = () => {
  // A simple SVG representing a chat interface
  const svgContent = `
<svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#ffffff" />
  
  <!-- Chat Interface Frame -->
  <rect x="50" y="50" width="700" height="400" rx="10" fill="#f8fafc" stroke="#e0e0e0" stroke-width="2" />
  
  <!-- Header -->
  <rect x="50" y="50" width="700" height="60" rx="10" fill="#4285f4" />
  <text x="100" y="85" font-family="Arial" font-size="18" fill="white" font-weight="bold">General Health Query</text>
  <text x="100" y="105" font-family="Arial" font-size="12" fill="white">AI-powered medical assistant</text>
  <circle cx="700" cy="80" r="20" fill="white" opacity="0.2" />
  <text x="700" y="85" font-family="Arial" font-size="14" fill="white" text-anchor="middle">👤</text>
  
  <!-- Chat Messages -->
  <!-- Bot message -->
  <rect x="70" y="130" width="300" height="70" rx="10" fill="#f0f2f5" />
  <text x="90" y="155" font-family="Arial" font-size="12" fill="#333333" width="260">
    I understand your headache symptoms. To help better:
  </text>
  <text x="90" y="175" font-family="Arial" font-size="12" fill="#333333" width="260">
    1. How long have you been experiencing these headaches?
  </text>
  <text x="90" y="190" font-family="Arial" font-size="10" fill="#777777">10:30 AM</text>
  
  <!-- User message -->
  <rect x="430" y="150" width="300" height="50" rx="10" fill="#4285f4" />
  <text x="450" y="175" font-family="Arial" font-size="12" fill="white" width="260">
    I've had this headache for about 3 days now.
  </text>
  <text x="710" y="190" font-family="Arial" font-size="10" fill="#777777" text-anchor="end">10:31 AM</text>
  
  <!-- Bot message -->
  <rect x="70" y="220" width="400" height="90" rx="10" fill="#f0f2f5" />
  <text x="90" y="245" font-family="Arial" font-size="12" fill="#333333" width="360">
    Thank you for that information. Since it's been persisting for 3 days, 
    that's considered a prolonged headache.
  </text>
  <text x="90" y="275" font-family="Arial" font-size="12" fill="#333333" width="360">
    Would you describe the pain as mild, moderate, or severe?
  </text>
  <text x="90" y="300" font-family="Arial" font-size="10" fill="#777777">10:32 AM</text>
  
  <!-- Quick Reply Options -->
  <rect x="70" y="330" width="80" height="30" rx="15" fill="#f0f2f5" stroke="#dee1e5" stroke-width="1" />
  <text x="110" y="350" font-family="Arial" font-size="12" fill="#333333" text-anchor="middle">Mild</text>
  
  <rect x="160" y="330" width="100" height="30" rx="15" fill="#f0f2f5" stroke="#dee1e5" stroke-width="1" />
  <text x="210" y="350" font-family="Arial" font-size="12" fill="#333333" text-anchor="middle">Moderate</text>
  
  <rect x="270" y="330" width="80" height="30" rx="15" fill="#f0f2f5" stroke="#dee1e5" stroke-width="1" />
  <text x="310" y="350" font-family="Arial" font-size="12" fill="#333333" text-anchor="middle">Severe</text>
  
  <!-- Input box -->
  <rect x="70" y="380" width="600" height="50" rx="25" fill="white" stroke="#dee1e5" stroke-width="1" />
  <text x="100" y="410" font-family="Arial" font-size="14" fill="#aaaaaa">Type your response...</text>
  <circle cx="650" cy="405" r="20" fill="#4285f4" />
  <text x="650" y="410" font-family="Arial" font-size="16" fill="white" text-anchor="middle">↑</text>
</svg>
  `;
  
  return Buffer.from(svgContent);
}

// Write the images to files
const outputDir = __dirname; // Current directory

// Create landing hero image
fs.writeFileSync(
  path.join(outputDir, 'landing-hero.svg'),
  createLandingHeroImage()
);

// Create demo screenshot image
fs.writeFileSync(
  path.join(outputDir, 'demo-screenshot.svg'),
  createDemoScreenshotImage()
);

console.log('Successfully created placeholder images:');
console.log('1. landing-hero.svg');
console.log('2. demo-screenshot.svg');
console.log('\nThese SVG images can be used directly in your React application.');
