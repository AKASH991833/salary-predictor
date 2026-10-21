module.exports.run = async (page, base, shot) => {
  const out = [];
  await page.waitForFunction(() => document.querySelector('#salary').textContent.includes('$'), {timeout: 10000});
  const sal = await page.$eval('#salary', e => e.textContent);
  out.push(['prediction rendered: ' + sal, /\$\d+k/.test(sal)]);
  const r2 = await page.$eval('#st-r2', e => parseFloat(e.textContent));
  out.push(['R2 in sane range: ' + r2, r2 > 0.5 && r2 <= 1]);
  const imp = await page.$$eval('.imp-row', r => r.length);
  out.push(['7 importance rows', imp === 7]);
  // change role and re-predict
  await page.select('#role', 'ml');
  await page.click('#predict');
  await new Promise(r=>setTimeout(r,300));
  const sal2 = await page.$eval('#salary', e => e.textContent);
  out.push(['re-predict after role change: ' + sal2, /\$\d+k/.test(sal2) && sal2 !== sal]);
  const rng = await page.$eval('#range', e => e.textContent);
  out.push(['range rendered: ' + rng, /\$\d+k - \$\d+k/.test(rng)]);
  if (shot) await page.screenshot({path: shot});
  return out;
};
