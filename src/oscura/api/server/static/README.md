# Oscura Web Dashboard - Static Assets

This directory contains static assets (CSS, JavaScript, images) for the Oscura web dashboard.

## Directory Structure

```
static/
├── css/          # Custom CSS stylesheets (optional)
├── js/           # Custom JavaScript files (optional)
└── img/          # Images and icons (optional)
```

## Usage

Static files are served automatically by FastAPI at `/static/` URL path.

To reference static assets in templates:

```html
<link rel="stylesheet" href="/static/css/custom.css">
<script src="/static/js/custom.js"></script>
<img src="/static/img/logo.png" alt="Logo">
```

## Built-in Assets

The dashboard uses CDN-hosted assets by default:

- **Bootstrap 5.3.0** - CSS framework
- **Bootstrap Icons 1.11.0** - Icon library
- **Plotly.js 2.26.0** - Interactive charting

Custom assets placed here will override or extend the default styling.
