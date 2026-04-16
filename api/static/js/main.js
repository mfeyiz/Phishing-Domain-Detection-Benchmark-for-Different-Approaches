const API_BASE = window.location.origin;

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const contentDiv = document.getElementById('resultContent');
    
    const isPhishing = data.is_phishing;
    const statusClass = isPhishing ? 'phishing' : 'clean';
    const statusText = isPhishing ? 'PHISHING' : 'TEMIZ';
    
    contentDiv.innerHTML = `
        <div class="result-item ${statusClass}">
            <span class="result-label">Tespit:</span>
            <span class="result-value ${statusClass}">${statusText}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Güven:</span>
            <span>${(data.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="result-item">
            <span class="result-label">Algoritma:</span>
            <span>${data.method}</span>
        </div>
    `;
    
    resultDiv.classList.remove('hidden');
}

async function analyze(method = null) {
    const original = document.getElementById('original').value.trim();
    const suspicious = document.getElementById('suspicious').value.trim();
    const threshold = parseFloat(document.getElementById('threshold').value) || 0.5;
    
    if (!original || !suspicious) {
        alert('Lütfen hem orijinal hem de şüpheli URL\'i girin');
        return;
    }
    
    showLoading();
    
    try {
        const payload = {
            original,
            suspicious,
            threshold
        };
        
        if (method) {
            payload.method = method;
        }
        
        const endpoint = method ? '/predict' : '/predict';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analiz başarısız');
        }
        
        const data = await response.json();
        displayResult(data);
        
    } catch (error) {
        alert('Hata: ' + error.message);
        hideLoading();
    }
}

async function analyzeAll() {
    const original = document.getElementById('original').value.trim();
    const suspicious = document.getElementById('suspicious').value.trim();
    const threshold = parseFloat(document.getElementById('threshold').value) || 0.5;
    
    if (!original || !suspicious) {
        alert('Lütfen hem orijinal hem de şüpheli URL\'i girin');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/predict/all', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ original, suspicious, threshold })
        });
        
        if (!response.ok) {
            throw new Error('Analiz başarısız');
        }
        
        const results = await response.json();
        
        let html = '';
        
        for (const [method, data] of Object.entries(results)) {
            if (data.error) continue;
            
            const isPhishing = data.is_phishing;
            const statusClass = isPhishing ? 'phishing' : 'clean';
            const statusText = isPhishing ? 'PHISHING' : 'TEMIZ';
            
            html += `
                <div class="result-item ${statusClass}">
                    <span class="result-label">${method.toUpperCase()}:</span>
                    <span class="result-value ${statusClass}">${statusText}</span>
                    <span>${(data.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
        }
        
        document.getElementById('resultContent').innerHTML = html;
        document.getElementById('result').classList.remove('hidden');
        
    } catch (error) {
        alert('Hata: ' + error.message);
        hideLoading();
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        
        if (!response.ok) {
            throw new Error('Geçmiş yüklenemedi');
        }
        
        const data = await response.json();
        const historyDiv = document.getElementById('historyList');
        
        if (data.history && data.history.length > 0) {
            historyDiv.innerHTML = data.history.map(item => `
                <div class="history-item" onclick="loadHistoryItem(${item.id})">
                    <div class="history-item-header">
                        <span>${new Date(item.created_at).toLocaleString('tr-TR')}</span>
                        <span class="history-item-method">${item.method}</span>
                    </div>
                    <div>${item.original.substring(0, 50)}...</div>
                </div>
            `).join('');
        } else {
            historyDiv.innerHTML = '<p>Henüz geçmiş yok</p>';
        }
        
    } catch (error) {
        console.error('Geçmiş yüklenirken hata:', error);
    }
}

async function loadHistoryItem(id) {
    try {
        const response = await fetch(`/history/${id}`);
        
        if (!response.ok) {
            throw new Error('Detay yüklenemedi');
        }
        
        const data = await response.json();
        
        document.getElementById('original').value = data.original;
        document.getElementById('suspicious').value = data.suspicious;
        document.getElementById('method').value = data.method;
        
        displayResult({
            is_phishing: data.is_phishing,
            confidence: data.confidence,
            method: data.method
        });
        
    } catch (error) {
        alert('Hata: ' + error.message);
    }
}

document.getElementById('analyzeBtn').addEventListener('click', () => {
    const method = document.getElementById('method').value;
    analyze(method);
});

document.getElementById('analyzeAllBtn').addEventListener('click', analyzeAll);

document.getElementById('loadHistory').addEventListener('click', loadHistory);