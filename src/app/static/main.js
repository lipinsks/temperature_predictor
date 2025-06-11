const form = document.getElementById('predict-form');
const btn = document.getElementById('predict-btn');
const spinner = document.getElementById('spinner');
const btnText = document.getElementById('btn-text');
const resultDiv = document.getElementById('result');

// Inicjalizacja mapy
const map = L.map('map').setView([51.759, 19.456], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);
let marker = L.marker([51.759, 19.456]).addTo(map);

// Obsługa kliknięcia na mapie
map.on('click', (e) => {
  const { lat, lng } = e.latlng;
  marker.setLatLng([lat, lng]);
  form.lat.value = lat.toFixed(6);
  form.lon.value = lng.toFixed(6);
});

// Lista gotowych miejsc
const places = {
  'Warsaw': [52.2297, 21.0122],
  'Krakow': [50.0647, 19.9450],
  'Berlin': [52.52, 13.405],
  'London': [51.5074, -0.1278],
  'New York': [40.7128, -74.0060]
};

document.getElementById('place').addEventListener('change', function() {
  const coords = places[this.value];
  if (coords) {
    const [lat, lon] = coords;
    map.setView([lat, lon], 8);
    marker.setLatLng([lat, lon]);
    form.lat.value = lat;
    form.lon.value = lon;
  }
});

// Obsługa formularza
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  btn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Loading';
  resultDiv.textContent = '';

  const lat = form.lat.value;
  const lon = form.lon.value;
  const timestamp = form.timestamp.value;
  const url = `/predict_temp?lat=${lat}&lon=${lon}&timestamp=${encodeURIComponent(timestamp)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {
      resultDiv.textContent = `Error: ${data.error}`;
    } else {
      const temp = parseFloat(data.temperature).toFixed(2);
      resultDiv.innerHTML = `Predicted: <span class="text-2xl">${temp}°C</span><br/><span class="text-sm text-gray-600">at ${data.timestamp}</span>`;
    }
  } catch (err) {
    resultDiv.textContent = `Error: ${err}`;
  } finally {
    spinner.classList.add('hidden');
    btn.disabled = false;
    btnText.textContent = 'Predict';
  }
});
