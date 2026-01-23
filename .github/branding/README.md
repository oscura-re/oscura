# Oscura Brand Assets

Official branding assets for the Oscura hardware reverse engineering framework.

## Logo Design

The Oscura logo features three layered waveforms representing multi-dimensional signal analysis - a core concept in hardware reverse engineering.

### Design Elements

- **Three stacked waveforms**: Representing multi-layer signal analysis
- **Cyan/aqua color palette**: Modern, technical aesthetic
- **Dark background**: Professional, sleek appearance
- **High contrast**: Excellent visibility across all contexts

### Color Palette

| Color          | Hex Code | Usage                   |
| -------------- | -------- | ----------------------- |
| Primary        | #0891B2  | Main cyan               |
| Highlight      | #22D3EE  | Bright cyan accents     |
| Accent         | #67E8F9  | Light cyan details      |
| Background     | #0C1415  | Deep teal-black         |
| Text           | #F8FAFC  | Off-white text          |
| Secondary Text | #94A3B8  | Muted gray for taglines |

## Files

### Vector (SVG) - Preferred

Use SVG files whenever possible for best quality at any size.

| File                  | Dimensions | Description                              |
| --------------------- | ---------- | ---------------------------------------- |
| `logo-icon.svg`       | 1024x1024  | Icon only, for avatars and small spaces  |
| `logo-full.svg`       | 800x1000   | Icon + "OSCURA" text below               |
| `logo-horizontal.svg` | 1200x400   | Icon left, text right (best for headers) |
| `logo-vertical.svg`   | 600x800    | Icon top, text bottom                    |
| `logo-wordmark.svg`   | 800x200    | "OSCURA" text only                       |
| `logo-dark.svg`       | 1024x1024  | Optimized for dark backgrounds           |
| `logo-light.svg`      | 1024x1024  | Optimized for light backgrounds          |
| `social-preview.svg`  | 1280x640   | GitHub social preview layout             |

### Raster (PNG) - GitHub Specific

| File                         | Dimensions | Purpose                              |
| ---------------------------- | ---------- | ------------------------------------ |
| `oscura-org-avatar.png`      | 512x512    | GitHub organization avatar           |
| `oscura-org-avatar-1024.png` | 1024x1024  | High-resolution avatar               |
| `oscura-repo-social.png`     | 1280x640   | Repository social preview (OG image) |
| `oscura-readme-header.png`   | 800x267    | README header image                  |

## Usage

### In README

```markdown
<p align="center">
  <img src=".github/branding/logo-horizontal.svg" alt="Oscura" width="600">
</p>
```

### In Documentation

```html
<img src="../.github/branding/logo-icon.svg" alt="Oscura" height="48" />
```

### As Favicon

Use `logo-icon.svg` or generate PNG from it at required sizes.

## GitHub Integration

### Organization Avatar

Upload `oscura-org-avatar.png` at:
https://github.com/organizations/oscura-re/settings/profile

### Repository Social Preview

Upload `oscura-repo-social.png` at:
https://github.com/oscura-re/oscura/settings

The social preview appears when the repository is shared on social media, Slack, Discord, etc.

## Guidelines

### Do

- Use vector (SVG) files when possible for best quality
- Maintain minimum size of 32px for icon visibility
- Use horizontal layout for headers and wide spaces
- Use icon-only for avatars, favicons, and small UI elements
- Keep clear space around logo (minimum 0.5x logo height)
- Use dark mode variant on dark backgrounds
- Use light mode variant on light backgrounds

### Don't

- Stretch or distort the logo
- Change the colors outside the defined palette
- Add effects like drop shadows or outlines
- Use low-resolution versions when SVG is available
- Place on busy or low-contrast backgrounds

## Technical Details

- **Font**: IBM Plex Sans (600 weight, 0.16em letter-spacing)
- **Fallback fonts**: Inter, -apple-system, sans-serif
- **SVG features**: Gaussian blur filters, linear gradients
- **Accessibility**: All SVGs include title and desc elements

## License

These brand assets are part of the Oscura project and follow the same MIT license.
