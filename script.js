const lenis = new Lenis({
  smoothWheel: true,
  lerp: 0.08,
  duration: 1.1,
  wheelMultiplier: 0.95,
});

function raf(time) {
  lenis.raf(time);
  requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  },
  { threshold: 0.18 }
);

document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));

const parallaxElements = document.querySelectorAll('.floating');
window.addEventListener('scroll', () => {
  const offset = window.scrollY;
  parallaxElements.forEach((el, index) => {
    const rate = (index + 1) * 0.05;
    el.style.transform = `translateY(${offset * rate * -1}px)`;
  });
});

const leadForm = document.querySelector('.lead-form');
leadForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  const button = leadForm.querySelector('button');
  if (!button) return;
  button.textContent = 'Request Received';
  button.setAttribute('disabled', 'true');
});
