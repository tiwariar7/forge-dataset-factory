/**
 * Arman Tiwari Signature Documentation Theme: "Forge Obsidian"
 * Interactive behaviors: Dark/Light Mode, Instant Search, Tabs, Code Copy.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initCodeCopy();
  initTabs();
  initSearch();
  initActiveSidebar();
});

// 1. Theme Management (Dark / Light)
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle-btn');
  const storedTheme = localStorage.getItem('forge-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', storedTheme);
  updateThemeIcon(storedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('forge-theme', next);
      updateThemeIcon(next);
    });
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  if (!icon) return;
  if (theme === 'light') {
    icon.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>`;
  } else {
    icon.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/></svg>`;
  }
}

// 2. Code Copy to Clipboard
function initCodeCopy() {
  document.querySelectorAll('.code-copy-btn, .pip-copy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const codeTarget = btn.closest('.code-window')?.querySelector('pre code') || 
                         btn.closest('.pip-install-box')?.querySelector('.pip-cmd');
      if (!codeTarget) return;

      const text = codeTarget.innerText.trim();
      try {
        await navigator.clipboard.writeText(text);
        const originalHTML = btn.innerHTML;
        btn.innerHTML = `<span style="color: var(--accent-emerald)">✓ Copied!</span>`;
        setTimeout(() => { btn.innerHTML = originalHTML; }, 2000);
      } catch (err) {
        console.error('Failed to copy', err);
      }
    });
  });
}

// 3. Tab Switcher
function initTabs() {
  document.querySelectorAll('.tab-container').forEach(container => {
    const buttons = container.querySelectorAll('.tab-button');
    const contents = container.querySelectorAll('.tab-content');

    buttons.forEach((btn, index) => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        if (contents[index]) contents[index].classList.add('active');
      });
    });
  });
}

// 4. Instant Search (Ctrl+K)
function initSearch() {
  const searchBtn = document.getElementById('search-trigger-btn');
  const modal = document.getElementById('search-modal-backdrop');
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');

  if (!modal || !input) return;

  function openSearch() {
    modal.style.display = 'flex';
    input.focus();
  }

  function closeSearch() {
    modal.style.display = 'none';
    input.value = '';
    if (results) results.innerHTML = '';
  }

  if (searchBtn) searchBtn.addEventListener('click', openSearch);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeSearch();
  });

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      modal.style.display === 'flex' ? closeSearch() : openSearch();
    }
    if (e.key === 'Escape' && modal.style.display === 'flex') {
      closeSearch();
    }
  });

  // Client-side index search
  const index = [
    { title: "Introduction & Overview", url: "index.html", snippet: "100% Local Document-to-SFT Dataset Factory." },
    { title: "Quickstart & Installation", url: "getting-started.html", snippet: "Install via pip and generate your first dataset in 3 lines." },
    { title: "Multi-Engine LLM Support", url: "engines.html", snippet: "Use llama.cpp (GGUF), Ollama, or vLLM / OpenAI-compatible backends." },
    { title: "Pipeline Architecture", url: "architecture.html", snippet: "Docling routing, Zero-Overlap Hybrid Chunker, and Quality Gate." },
    { title: "API Reference", url: "api-reference.html", snippet: "IngestionRouter, HybridChunker, SFTGenerator, QualityGate, Exporter." },
    { title: "Examples & Cookbooks", url: "examples.html", snippet: "Processing 10-K financial reports, spreadsheets, codebases, and slides." },
  ];

  if (input) {
    input.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) {
        results.innerHTML = '';
        return;
      }
      const matches = index.filter(item => 
        item.title.toLowerCase().includes(q) || item.snippet.toLowerCase().includes(q)
      );

      if (matches.length === 0) {
        results.innerHTML = `<div style="padding: 1.5rem; text-align: center; color: var(--text-muted);">No results found for "${q}"</div>`;
      } else {
        results.innerHTML = matches.map(m => `
          <a href="${m.url}" class="search-result-item">
            <div class="search-result-title">${m.title}</div>
            <div class="search-result-snippet">${m.snippet}</div>
          </a>
        `).join('');
      }
    });
  }
}

// 5. Active Sidebar Navigation Highlight
function initActiveSidebar() {
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.sidebar-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || (currentPath === '' && href === 'index.html')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

