// Mouse follower circle effect
(function () {
  // Only run on devices with mouse (not touch devices)
  // if ('ontouchstart' in window || navigator.maxTouchPoints > 0) {
  //   return;
  // }

  // Function to create the cursor if it doesn't exist
  function initCursor() {
    if (document.querySelector(".mouse-follower")) return;
    const cursor = document.createElement("div");
    cursor.className = "mouse-follower";
    document.body.appendChild(cursor);
    cursor.style.opacity = "1";
  }

  initCursor(); // Initial creation
  document.addEventListener("astro:page-load", initCursor); // Re-create on page change

  let mouseX = 0;
  let mouseY = 0;
  let cursorX = 0;
  let cursorY = 0;
  const speed = 0.15; // Lower value = slower, smoother movement

  // Track mouse position
  document.addEventListener("mousemove", e => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  // Animate cursor with smooth delay
  function animate() {
    const cursor = document.querySelector(".mouse-follower"); // Get fresh reference
    if (cursor) {
      // Calculate distance
      const distX = mouseX - cursorX;
      const distY = mouseY - cursorY;

      // Update cursor position with easing
      cursorX += distX * speed;
      cursorY += distY * speed;

      // Apply transform
      cursor.style.transform = `translate(${cursorX}px, ${cursorY}px)`;
    }
    requestAnimationFrame(animate);
  }

  animate();

  // Add scale effect on click
  document.addEventListener("mousedown", () => {
    const cursor = document.querySelector(".mouse-follower");
    if (cursor) {
      cursor.style.transform = `translate(${cursorX}px, ${cursorY}px) scale(0.8)`;
    }
  });

  document.addEventListener("mouseup", () => {
    const cursor = document.querySelector(".mouse-follower");
    if (cursor) {
      cursor.style.transform = `translate(${cursorX}px, ${cursorY}px) scale(1)`;
    }
  });

  // Hide cursor when leaving the window
  document.addEventListener("mouseleave", () => {
    const cursor = document.querySelector(".mouse-follower");
    if (cursor) {
      cursor.style.opacity = "0";
    }
  });

  document.addEventListener("mouseenter", () => {
    const cursor = document.querySelector(".mouse-follower");
    if (cursor) {
      cursor.style.opacity = "1";
    }
  });
})();