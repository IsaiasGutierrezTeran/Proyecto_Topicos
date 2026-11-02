const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
let app;
let response;
const requests = [];
vm.runInNewContext(script, {
  Vue: {ref: value => ({value}), computed: fn => ({get value() {return fn();}}), createApp: options => ({mount() {app = options.setup();}})},
  fetch: async (url, options) => {requests.push({url, options}); if(response instanceof Error) throw response; return response(url);}
});
(async () => {
  const metrics = JSON.parse(fs.readFileSync('artifacts/metricas.json', 'utf8'));
  response = () => ({ok:true, json:async()=>metrics});
  await app.cargarMetricas();
  assert.equal(app.incompletos.value, false);
  assert.equal(app.matrizValida.value, true);
  assert.deepEqual(Array.from(app.clases.value), metrics.clases);
  assert.match(app.porcentaje(metrics.accuracy_test), /59[.,]44%/);
  assert.equal(app.porcentaje(null), 'No disponible');
  assert.equal(app.porcentaje(0), '0,00%');
  assert.equal(app.reporte('neutro').support, 60);
  app.metricas.value = {};
  assert.equal(app.incompletos.value, true);
  assert.equal(app.matrizValida.value, false);
  app.metricas.value = {...metrics, clases:['positivo','negativo','neutro']};
  assert.equal(app.clases.value[0], 'positivo');
  response = () => ({ok:false});
  await app.cargarMetricas();
  assert.ok(app.mensajeMetricas.value);
  assert.equal(app.cargandoMetricas.value, false);
  response = () => ({ok:true,json:async()=>null});
  await app.cargarMetricas();
  assert.match(app.mensajeMetricas.value, /inválidos/);
  response = new Error('Network error');
  await app.cargarMetricas();
  assert.ok(app.mensajeMetricas.value);
  assert.equal(app.cargandoMetricas.value, false);
  let finish;
  response = () => new Promise(resolve => {finish=resolve;});
  const loading = app.cargarMetricas();
  assert.equal(app.cargandoMetricas.value, true);
  const count = requests.length;
  await app.cargarMetricas();
  assert.equal(requests.length, count);
  finish({ok:true,json:async()=>metrics});
  await loading;
  app.texto.value = 'Excelente producto';
  response = url => ({ok:true,status:200,json:async()=>url === '/preprocesar' ? {normalizado:'excelente producto',tokens:['excelente','producto'],sin_stopwords:['excelente','producto'],stemming:[],lematizacion:[]} : {sentimiento:'positivo',probabilidades:{positivo:0.8,neutro:0.1,negativo:0.1}}});
  await app.analizar();
  assert.equal(app.error.value, '');
  assert.equal(app.prediccion.value.sentimiento, 'positivo');
  assert.equal(app.procesamiento.value.tokens.length, 2);
  for(const request of requests.slice(-2)) {
    assert.equal(request.options.method, 'POST');
    assert.equal(JSON.parse(request.options.body).texto, app.texto.value);
  }
  console.log('OK: métricas reales, formatos, datos incompletos, orden de clases, errores, carga y flujo de reseñas con API simulada.');
})().catch(error => {console.error(error);process.exitCode=1;});
