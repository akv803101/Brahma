/* ─────────────────────────────────────────────────────────────────────── */
/* BRAHMA DASHBOARD — Interactivity                                        */
/* ─────────────────────────────────────────────────────────────────────── */

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Remove active state from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Add active state to clicked button
    event.target.classList.add('active');
}

// Load leaderboard from CSV
function loadLeaderboard() {
    fetch('../outputs/data/leaderboard.csv')
        .then(response => response.text())
        .then(data => {
            const rows = data.trim().split('\n');
            const headers = rows[0].split(',');
            
            // Create table HTML
            let tableHTML = '<table><thead><tr>';
            for (let header of headers) {
                tableHTML += `<th>${header.trim()}</th>`;
            }
            tableHTML += '</tr></thead><tbody>';

            // Add data rows
            for (let i = 1; i < rows.length; i++) {
                const cols = rows[i].split(',');
                tableHTML += '<tr>';
                for (let j = 0; j < cols.length; j++) {
                    const value = cols[j].trim();
                    // Highlight best model (first row)
                    const cellClass = i === 1 ? 'class="best-model"' : '';
                    tableHTML += `<td ${cellClass}>${value}</td>`;
                }
                tableHTML += '</tr>';
            }
            tableHTML += '</tbody></table>';

            // Insert into page
            document.getElementById('leaderboard').innerHTML = tableHTML;

            // Extract and display top model metrics
            if (rows.length > 1) {
                const topRowData = rows[1].split(',');
                if (headers.includes('Model') && headers.includes('ROC-AUC')) {
                    const modelIdx = headers.indexOf('Model');
                    const rocIdx = headers.indexOf('ROC-AUC');
                    document.getElementById('best-model').textContent = topRowData[modelIdx]?.trim() || '—';
                    document.getElementById('roc-auc').textContent = topRowData[rocIdx]?.trim() || '—';
                }
            }
        })
        .catch(err => {
            console.warn('Could not load leaderboard:', err);
            document.getElementById('leaderboard').innerHTML = 
                '<p style="color: var(--text-light);">Leaderboard data not available yet.</p>';
        });
}

// Load dataset metadata from drift config
function loadMetadata() {
    fetch('../outputs/data/drift_config.json')
        .then(response => response.json())
        .then(data => {
            // Set metrics if available
            if (data.n_samples) {
                document.getElementById('total-records').textContent = 
                    data.n_samples.toLocaleString();
            }
            if (data.target_distribution) {
                const churnRate = (data.target_distribution['1'] || 0) * 100;
                document.getElementById('churn-rate').textContent = 
                    churnRate.toFixed(1) + '%';
            }
        })
        .catch(err => {
            console.warn('Could not load metadata:', err);
        });
}

// Set timestamp
function setTimestamp() {
    const now = new Date();
    const dateStr = now.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZone: 'UTC'
    });
    
    document.getElementById('timestamp').textContent = `Generated: ${dateStr} UTC`;
    document.getElementById('footer-time').textContent = dateStr;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimestamp();
    loadMetadata();
    loadLeaderboard();
});

// Highlight best model row in leaderboard
const style = document.createElement('style');
style.textContent = `
    table tbody td.best-model {
        background-color: #DBEAFE;
        font-weight: 600;
        color: var(--primary);
    }
`;
document.head.appendChild(style);
