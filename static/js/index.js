// ===== TechyVia JavaScript ===== //

// Global variables
let currentTestimonial = 2;
let particles = [];
let canvas, ctx;

// Testimonials data
const testimonials = [
  {
    text: "TechyVia's Python tutorials helped me transition from a business analyst to a data engineer in just 6 months. The clear explanations and practical examples made all the difference.",
    author: "Soundarya",
    role: "Data Engineer",
    avatar: "SJ"
  },
  {
    text: "The SQL resources on TechyVia are exceptional. I've significantly improved my query optimization skills and database design knowledge thanks to their detailed guides.",
    author: "Krishna",
    role: "Database Administrator",
    avatar: "KR"
  },
  {
    text: "From zero to hero in machine learning! The structured learning path and mentorship program accelerated my career beyond expectations.",
    author: "Priya Patel",
    role: "Data Scientist at Google",
    avatar: "PP"
  }
];

// ===== Particle System ===== //
class Particle {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.vx = (Math.random() - 0.5) * 0.5;
    this.vy = (Math.random() - 0.5) * 0.5;
    this.size = Math.random() * 2 + 1;
    this.life = 1.0;
    this.decay = Math.random() * 0.01 + 0.005;
    this.color = `hsl(${Math.random() * 60 + 180}, 70%, 60%)`;
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.life -= this.decay;
    
    // Wrap around screen edges
    if (this.x < 0) this.x = canvas.width;
    if (this.x > canvas.width) this.x = 0;
    if (this.y < 0) this.y = canvas.height;
    if (this.y > canvas.height) this.y = 0;
  }

  draw() {
    ctx.globalAlpha = this.life * 0.6;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
    
    // Add glow effect
    ctx.shadowBlur = 20;
    ctx.shadowColor = this.color;
    ctx.globalAlpha = this.life * 0.3;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size * 2, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  }

  isDead() {
    return this.life <= 0;
  }
}

function initParticles() {
  canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  
  ctx = canvas.getContext('2d');
  resizeCanvas();
  
  // Create initial particles
  for (let i = 0; i < 50; i++) {
    particles.push(new Particle(
      Math.random() * canvas.width,
      Math.random() * canvas.height
    ));
  }
  
  animateParticles();
}

function resizeCanvas() {
  if (!canvas) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function animateParticles() {
  if (!canvas || !ctx) return;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Update and draw particles
  for (let i = particles.length - 1; i >= 0; i--) {
    particles[i].update();
    particles[i].draw();
    
    if (particles[i].isDead()) {
      particles.splice(i, 1);
    }
  }
  
  // Add new particles occasionally
  if (particles.length < 50 && Math.random() < 0.1) {
    particles.push(new Particle(
      Math.random() * canvas.width,
      Math.random() * canvas.height
    ));
  }
  
  requestAnimationFrame(animateParticles);
}

// ===== Navigation Functions ===== //
function initNavigation() {
  const navbar = document.getElementById('navbar');
  const navToggle = document.getElementById('nav-toggle');
  const navMenu = document.getElementById('nav-menu');
  
  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    updateScrollProgress();
  });
  
  // Mobile menu toggle
  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      navToggle.classList.toggle('active');
      navMenu.classList.toggle('active');
    });
    
    // Close mobile menu when clicking on links
    navMenu.addEventListener('click', (e) => {
      if (e.target.tagName === 'A') {
        navToggle.classList.remove('active');
        navMenu.classList.remove('active');
      }
    });
  }
}

function updateScrollProgress() {
  const scrollProgress = document.querySelector('.scroll-progress-bar');
  if (!scrollProgress) return;
  
  const windowHeight = window.innerHeight;
  const documentHeight = document.documentElement.scrollHeight - windowHeight;
  const scrolled = window.scrollY;
  const progress = (scrolled / documentHeight) * 100;
  
  scrollProgress.style.width = `${Math.min(progress, 100)}%`;
}

// ===== Smooth Scrolling ===== //
function scrollToSection(sectionId) {
  const element = document.getElementById(sectionId);
  if (element) {
    const navbarHeight = document.querySelector('.navbar').offsetHeight;
    const elementPosition = element.offsetTop - navbarHeight - 20;
    
    window.scrollTo({
      top: elementPosition,
      behavior: 'smooth'
    });
  }
}

// ===== Animation on Scroll ===== //
function initScrollAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);
  
  // Observe all elements with animate-on-scroll class
  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    observer.observe(el);
  });
}

// ===== Code Editor Functions ===== //
function initCodeEditor() {
  const runButton = document.getElementById('run-code');
  const codeEditor = document.getElementById('code-editor');
  const codeOutput = document.getElementById('code-output');
  
  if (!runButton || !codeEditor || !codeOutput) return;
  
  runButton.addEventListener('click', () => {
    const code = codeEditor.value;
    
    // Add loading effect
    runButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
    runButton.disabled = true;
    
    // Simulate code execution (replace with actual execution logic)
    setTimeout(() => {
      try {
        // This is a mock execution - in a real implementation,
        // you would send the code to a backend service
        const output = mockPythonExecution(code);
        codeOutput.textContent = output;
      } catch (error) {
        codeOutput.textContent = `Error: ${error.message}`;
      }
      
      // Reset button
      runButton.innerHTML = '<i class="fas fa-play"></i> Run Code';
      runButton.disabled = false;
    }, 1500);
  });
}

function mockPythonExecution(code) {
  // Mock Python execution for demo purposes
  if (code.includes('print("Hello')) {
    return 'Hello, Future Data Engineer! Welcome to TechyVia\nOriginal: [1, 4, 9, 16, 25]\nSquare roots: [1.0, 2.0, 3.0, 4.0, 5.0]';
  } else if (code.includes('def greet')) {
    return 'Hello, Future Data Engineer! Welcome to TechyVia\nOriginal: [1, 4, 9, 16, 25]\nSquare roots: [1.0, 2.0, 3.0, 4.0, 5.0]';
  } else {
    return 'Code executed successfully!\n\nTry the default example or write your own Python code.';
  }
}

// ===== Testimonials Functions ===== //
function showTestimonial(index) {
  if (index < 0 || index >= testimonials.length) return;
  
  currentTestimonial = index;
  const testimonial = testimonials[index];
  
  // Update content
  const textElement = document.getElementById('testimonial-text');
  const nameElement = document.getElementById('author-name');
  const roleElement = document.getElementById('author-role');
  const avatarElement = document.getElementById('author-avatar');
  
  if (textElement) textElement.textContent = testimonial.text;
  if (nameElement) nameElement.textContent = testimonial.author;
  if (roleElement) roleElement.textContent = testimonial.role;
  if (avatarElement) avatarElement.textContent = testimonial.avatar;
  
  // Update dots
  document.querySelectorAll('.testimonial-dots .dot').forEach((dot, i) => {
    dot.classList.toggle('active', i === index);
  });
}

function initTestimonials() {
  // Auto-rotate testimonials
  setInterval(() => {
    currentTestimonial = (currentTestimonial + 1) % testimonials.length;
    showTestimonial(currentTestimonial);
  }, 5000);
}

// ===== Contact Form ===== //
function initContactForm() {
  const contactForm = document.getElementById('contact-form');
  if (!contactForm) return;
  
  contactForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const formData = new FormData(contactForm);
    const submitButton = contactForm.querySelector('button[type="submit"]');
    
    // Add loading effect
    const originalText = submitButton.innerHTML;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    submitButton.disabled = true;
    
    // Simulate form submission
    setTimeout(() => {
      // Reset form
      contactForm.reset();
      
      // Show success message
      showNotification('Message sent successfully! We\'ll get back to you soon.', 'success');
      
      // Reset button
      submitButton.innerHTML = originalText;
      submitButton.disabled = false;
    }, 2000);
  });
}

// ===== Notifications ===== //
function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.style.cssText = `
    position: fixed;
    top: 100px;
    right: 20px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: var(--text-primary);
    z-index: 10000;
    transform: translateX(400px);
    transition: transform 0.3s ease;
    max-width: 400px;
  `;
  
  if (type === 'success') {
    notification.style.borderColor = 'var(--secondary)';
  } else if (type === 'error') {
    notification.style.borderColor = '#EF4444';
  }
  
  notification.innerHTML = `
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
      <span>${message}</span>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Animate in
  setTimeout(() => {
    notification.style.transform = 'translateX(0)';
  }, 100);
  
  // Remove after delay
  setTimeout(() => {
    notification.style.transform = 'translateX(400px)';
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 4000);
}

// ===== Back to Top Button ===== //
function initBackToTop() {
  const backToTopButton = document.getElementById('back-to-top');
  if (!backToTopButton) return;
  
  window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
      backToTopButton.classList.add('visible');
    } else {
      backToTopButton.classList.remove('visible');
    }
  });
  
  backToTopButton.addEventListener('click', () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
}

// ===== Utility Functions ===== //
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function throttle(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
}

// ===== Keyboard Navigation ===== //
function initKeyboardNavigation() {
  document.addEventListener('keydown', (e) => {
    // ESC key closes mobile menu
    if (e.key === 'Escape') {
      const navToggle = document.getElementById('nav-toggle');
      const navMenu = document.getElementById('nav-menu');
      
      if (navToggle && navMenu && navMenu.classList.contains('active')) {
        navToggle.classList.remove('active');
        navMenu.classList.remove('active');
      }
    }
    
    // Arrow keys for testimonials
    if (e.key === 'ArrowLeft') {
      const newIndex = (currentTestimonial - 1 + testimonials.length) % testimonials.length;
      showTestimonial(newIndex);
    } else if (e.key === 'ArrowRight') {
      const newIndex = (currentTestimonial + 1) % testimonials.length;
      showTestimonial(newIndex);
    }
  });
}

// ===== Performance Monitoring ===== //
function initPerformanceMonitoring() {
  // Monitor page load performance
  window.addEventListener('load', () => {
    if ('performance' in window) {
      const loadTime = performance.now();
      console.log(`TechyVia loaded in ${loadTime.toFixed(2)}ms`);
    }
  });
  
  // Monitor scroll performance
  let isScrolling = false;
  const optimizedScroll = throttle(() => {
    updateScrollProgress();
  }, 16); // 60fps
  
  window.addEventListener('scroll', optimizedScroll);
}

// ===== Accessibility Enhancements ===== //
function initAccessibility() {
  // Add focus indicators for keyboard navigation
  const focusableElements = document.querySelectorAll('a, button, input, textarea, select');
  
  focusableElements.forEach(element => {
    element.addEventListener('focus', () => {
      element.style.outline = '2px solid var(--primary)';
      element.style.outlineOffset = '2px';
    });
    
    element.addEventListener('blur', () => {
      element.style.outline = '';
      element.style.outlineOffset = '';
    });
  });
  
  // Announce page changes for screen readers
  const announcePageChange = (message) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.style.position = 'absolute';
    announcement.style.left = '-10000px';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  };
}

// ===== Error Handling ===== //
function initErrorHandling() {
  window.addEventListener('error', (e) => {
    console.error('TechyVia Error:', e.error);
    
    // Show user-friendly error message for critical errors
    if (e.error && e.error.stack && e.error.stack.includes('critical')) {
      showNotification('Something went wrong. Please refresh the page.', 'error');
    }
  });
  
  window.addEventListener('unhandledrejection', (e) => {
    console.error('TechyVia Promise Rejection:', e.reason);
  });
}

// ===== Main Initialization ===== //
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 TechyVia Initializing...');
  
  try {
    // Initialize all modules
    initParticles();
    initNavigation();
    initScrollAnimations();
    initCodeEditor();
    initTestimonials();
    initContactForm();
    initBackToTop();
    initKeyboardNavigation();
    initPerformanceMonitoring();
    initAccessibility();
    initErrorHandling();
    
    console.log('✅ TechyVia Loaded Successfully');
    
    // Show initial testimonial
    showTestimonial(currentTestimonial);
    
  } catch (error) {
    console.error('❌ TechyVia Initialization Error:', error);
    showNotification('Failed to initialize some features. Please refresh the page.', 'error');
  }
});

// ===== Window Event Listeners ===== //
window.addEventListener('resize', debounce(() => {
  resizeCanvas();
}, 250));

window.addEventListener('beforeunload', () => {
  // Cleanup
  particles = [];
  if (canvas && ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
});

// ===== Export Functions for Global Access ===== //
window.TechyVia = {
  scrollToSection,
  showTestimonial,
  showNotification
};