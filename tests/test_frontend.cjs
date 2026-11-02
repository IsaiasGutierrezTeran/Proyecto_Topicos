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
  for (const texto of ['', '  ', 'ab', 'a'.repeat(5001)]) {
    app.texto.value = texto;
    const antes = requests.length;
    await app.analizar();
    assert.ok(app.error.value);
    assert.equal(requests.length, antes);
  }
  for (const ejemplo of app.ejemplos) {
    assert.ok(ejemplo.texto.length >= 3 && ejemplo.texto.length <= 5000);
  }
  app.procesamiento.value = {tokens:['no', 'me', 'gusta', 'me'], sin_stopwords:['no', 'gusta']};
  assert.deepEqual(Array.from(app.eliminadas.value), ['me', 'me']);
  assert.equal(app.etapasNlp.value.length, 6);
  app.procesamiento.value = {};
  assert.equal(app.eliminadas.value, null);
  assert.equal(app.etapasNlp.value[0].valor, 'No disponible');
  response = () => ({ok:true, json:async()=>metrics});
  await app.cargarMetricas();
  assert.equal(app.incompletos.value, false);
  assert.equal(app.matrizValida.value, true);
  assert.deepEqual(Array.from(app.clases.value), metrics.clases);
  // El resultado esperado depende del artefacto vigente, no de un entrenamiento anterior.
  const claves = ['accuracy_test', 'precision_macro_test', 'recall_macro_test',
    'f1_macro_test', 'f1_macro_cv', 'f1_macro_baseline_mayoritaria'];
  for (const clave of claves) {
    const valor = metrics[clave];
    assert.equal(typeof valor, 'number', `${clave} debe ser numérico`);
    assert.ok(Number.isFinite(valor) && valor >= 0 && valor <= 1, `${clave} fuera de rango`);
    const esperado = (valor * 100).toFixed(2).replace('.', ',') + '%';
    const resultado = app.porcentaje(valor);
    assert.match(resultado, /^\d{1,3},\d{2}%$/, `${clave}: formato español con dos decimales`);
    assert.equal(resultado, esperado, `${clave}: conversión y redondeo`);
  }
  // Datos sintéticos solo para verificar el formateador, no métricas del modelo.
  for (const [valor, esperado] of [[0.123456, '12,35%'], [0.5, '50,00%'], [1, '100,00%']]) {
    assert.equal(app.porcentaje(valor), esperado);
  }
  for (const valor of [undefined, NaN, Infinity, -0.1, 1.1, '0.5']) {
    assert.equal(app.porcentaje(valor), 'No disponible');
  }
  assert.equal(app.porcentaje(null), 'No disponible');
  assert.equal(app.porcentaje(0), '0,00%');
  for (const clase of metrics.clases) {
    assert.equal(app.reporte(clase).support, metrics.reporte_clasificacion[clase].support);
  }
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
