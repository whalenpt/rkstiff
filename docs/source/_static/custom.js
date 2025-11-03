document.addEventListener("DOMContentLoaded", function() {
    // Find the API Reference link in the sidebar
    var apiLink = document.querySelector('.toctree-l1 a[href*="api.html"]');
    
    if (apiLink) {
        // Find its parent li element
        var apiLi = apiLink.closest('.toctree-l1');
        
        if (apiLi) {
            // Add the 'current' class to make it expanded
            apiLi.classList.add('current');
            
            // Find any nested ul and make sure it's visible
            var nestedUl = apiLi.querySelector('ul');
            if (nestedUl) {
                nestedUl.style.display = 'block';
            }
        }
    }
});