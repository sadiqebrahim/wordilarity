 const TARGET = "SYNTHETIC"; 
    let guesses = [];
    let attempts = 0;
    const MAX_ATTEMPTS = 10;

    const input = document.getElementById('user-input');
    const container = document.getElementById('list-container');
    const attemptDisplay = document.getElementById('attempt-counter');
    const highDisplay = document.getElementById('high-score');
    const sid = document.getElementById('sid');
    const el = document.getElementById("target-hint");
    const noWord = document.getElementById("no-word");
    const hint1 = document.getElementById("hint1");
    const hint2 = document.getElementById("hint2");
    const hint3 = document.getElementById("hint3");
    const sidebar = document.getElementById("sidebar");

    sid.innerText = "#" + Math.floor(Math.random() * 9000 + 1000);


      document.addEventListener("DOMContentLoaded", function () {
            myFunction();
        });

        async function myFunction() {
            console.log("Page loaded or refreshed!");
               const response = await fetch('http://localhost:5000/set_target', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            });
            if (!response.ok) return 0;
            const data = await response.json();
            return data;
            
        }

    hint1.addEventListener("click", async () => {
        hint1.innerText = await get_hint1();
        hint1.classList.remove("hazy");
        const stathin2 = document.getElementById("stathin2");
        stathin2.classList.remove("hidden");
        });
    hint2.addEventListener("click", async () => {
        hint2.innerText = await get_hint2();
        hint2.classList.remove("hazy");
        const stathin3 = document.getElementById("stathin3");
        stathin3.classList.remove("hidden");
        });
    hint3.addEventListener("click", async () => {
        hint3.innerText = await get_target();
        hint3.classList.remove("hazy");
        });

    function updateDisplay() {
        // Sort guesses by similarity score descending
        guesses.sort((a, b) => b.score - a.score);

        // Clear and rebuild (In a complex app, we'd use a virtual DOM or FLIP)
        container.innerHTML = '';
        guesses.forEach((g, index) => {
            const row = document.createElement('div');
            row.className = 'guess-row';
            row.style.borderLeft = `4px solid ${getColor(g.score)}`;
            
            row.innerHTML = `
                <div class="word-tag">${g.word}</div>
                <div class="score-tag" style="color:${getColor(g.score)}">${g.score}</div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: ${g.score * 100}%; background: ${getColor(g.score)}"></div>
                </div>
            `;
            container.appendChild(row);
        });

        // Update Stats
        if(guesses.length > 0) {
            highDisplay.innerText = guesses[0].score;
        }
        attemptDisplay.innerText = `${attempts} / ${MAX_ATTEMPTS}`;
    }

    function getColor(score) {
        if (score > 0.8) return '#3fb950'; // Green
        if (score > 0.5) return '#ff9a00'; // Orange
        if (score > 0.2) return '#9d50bb'; // Purple
        return '#58a6ff'; // Blue
    }

    input.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
        const val = input.value.toLowerCase().trim();
        if (!val || attempts >= MAX_ATTEMPTS) return;

        // Check if word is "meaningful" (exists in our Python dictionary)
        const valid = await isValidWord(val);
        if (!valid) {
            showError()
            input.value = '';
            return;
        }

        const score = await calculateSimilarity(val);
        attempts++;
        guesses.push({ word: val.toUpperCase(), score: score.toFixed(3) });
        
        input.value = '';
        updateDisplay();

        if (score >= 0.999) { // Using threshold for float precision
            const tar = await get_target()
            console.log(tar)
            el.innerHTML = el.innerHTML.replace(
                "[ENCRYPTED]",
                `<span class="decrypted-text">${tar}</span>`
                );
            showEndScreen("WINNER!!!", "win-glow");
        } else if (attempts >= MAX_ATTEMPTS) {
            const tar = await get_target()
            el.innerHTML = el.innerHTML.replace(
                "[ENCRYPTED]",
                `<span class="decrypted-text">${tar}</span>`
                );
            showEndScreen("LAVDENA BHOJYAM", "fail-glow");
        }
    }
});


    function showEndScreen(text, className) {
        const overlay = document.getElementById('overlay');
        overlay.innerHTML = `<h1 class="${className}">${text}</h1><button onclick="location.reload()" style="background:var(--neon-blue); border:none; padding:10px 20px; cursor:pointer;">RESTART</button>`;
        overlay.style.display = 'block';
        input.disabled = true;
    }


async function isValidWord(word) {
    const response = await fetch('http://localhost:5000/check_word', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: word })
    });
    const data = await response.json();
    return data.valid;
}

async function calculateSimilarity(word) {
    const response = await fetch('http://localhost:5000/get_similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: word })
    });
    if (!response.ok) return 0;
    const data = await response.json();
    return data.score; // Returns the float from Python
}

async function get_target() {
    const response = await fetch('http://localhost:5000/get_target', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return 0;
    const data = await response.json();
    return data.target; // Returns the float from Python
}


async function get_hint1() {
    const response = await fetch('http://localhost:5000/get_hint1', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return 0;
    const data = await response.json();
    return data.hint; // Returns the float from Python
}

async function get_hint2() {
    const response = await fetch('http://localhost:5000/get_hint2', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return 0;
    const data = await response.json();
    return data.hint; // Returns the float from Python
}


function showError() {
    // show instantly
    noWord.classList.add("visible");

    // after 2 seconds → start fade out
    setTimeout(() => {
        noWord.style.opacity = "0";

        // after fade completes → fully hide
        setTimeout(() => {
            noWord.classList.remove("visible");
            noWord.style.opacity = ""; // reset
        }, 400); // match CSS transition time
    }, 1000);
}